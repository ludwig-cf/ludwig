/*****************************************************************************
 *
 *  phi_stats.c
 *
 *  Order parameter statistics.
 *  There is a general version using a compensated sum for the global
 *  total of each scalar (sensitive to threads/order in some cases).
 *
 *  The variance is always computed in a single sweep form; we don't
 *  care too much about the exact result, so this is just a standard
 *  floating point sum.
 *
 *  There is also a version which adds a correction for BBL
 *  (specifically for the case of binary fluid). This might
 *  be updated and relocated.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "phi_stats.h"
#include "kernel.h"
#include "util_sum.h"

/* For internal use in accumulating sum for one scalar order parameter */

typedef struct sum_s sum_t;
struct sum_s {
  klein_t qsum;             /* Global sum sensitive to round-off ... */
  double  qvar;             /* Ditto, but don't care for variance. */
  double  qmin;             /* minimum */
  double  qmax;             /* maximum */
  double  vol;              /* Volume is the same in all cases */
};

int stats_field_q_reduce(field_t * field, map_t * map, int nxx, sum_t * sum,
			 int rank, MPI_Comm comm);

int stats_field_reduce(field_t * obj, map_t * map, double * fmin,
		       double * fmax, double * fsum, double * fvar,
		       double * fvol, int rank, MPI_Comm comm);
int stats_field_local(field_t * obj, map_t * map, double * fmin, double * fmax,
		      double * fsum, double * fvar, double * fvol);


__global__ void stats_field_q_kernel(kernel_3d_t k3d, field_t * q,
				     map_t * map, int nxx, sum_t * sum);

/*****************************************************************************
 *
 *  Utilities
 *
 *****************************************************************************/

__host__ __device__ static inline double double_min(double x, double y) {
  return (x < y) ? x : y;
}

__host__ __device__ static inline double double_max(double x, double y) {
  return (x > y) ? x : y;
}

__host__ __device__ static inline sum_t sum_zero() {
  sum_t sum = {
    .qsum = klein_zero(),
    .qvar = 0.0,
    .qmin = +DBL_MAX,
    .qmax = -DBL_MAX,
    .vol  = 0.0
  };
  return sum;
}

/*****************************************************************************
 *
 *  stats_field_info
 *
 *****************************************************************************/

int stats_field_info(field_t * field, map_t * map) {

  MPI_Comm comm = MPI_COMM_NULL;

  /* Labelling */
  const char * q1[5] = {"phi", "phi", "phi", "phi", "phi"}; /* default */
  const char * q3[3] = {"Px ", "Py ", "Pz "};
  const char * q5[5] = {"Qxx", "Qxy", "Qxz", "Qyy", "Qyz"};
  const char ** q    = NULL;

  assert(field);
  assert(map);

  switch (field->nf) {
  case 3:
    q = q3;
    break;
  case 5:
    q = q5;
    break;
  default:
    q = q1;
  }

  pe_mpi_comm(field->pe, &comm);

  for (int n = 0; n < field->nf; n++) {

    sum_t sum = {0};
    stats_field_q_reduce(field, map, n, &sum, 0, comm);

    {
      double qsum  = klein_sum(&sum.qsum);
      double rvol = 1.0/sum.vol;
      double qbar = rvol*qsum;                      /* mean */
      double qvar = rvol*sum.qvar  - qbar*qbar;     /* variance */

      pe_info(field->pe, "[%3s] %14.7e %14.7e %14.7e %14.7e %14.7e\n",
	      q[n], qsum, qbar, qvar, sum.qmin, sum.qmax);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_q_reduce
 *
 *  This is a global reduction to rank in communicator comm.
 *
 *  We expect and assert NQAB to be the largest number of field elements
 *  to avoid memory allocation and deallocation here.
 *
 *****************************************************************************/

int stats_field_q_reduce(field_t * field, map_t * map, int nxx, sum_t * sum,
			 int rank, MPI_Comm comm) {

  int nlocal[3] = {0};
  sum_t sum_local = sum_zero();  /* host copy */
  sum_t * dsum = NULL;           /* device copy */

  assert(field);
  assert(map);

  cs_nlocal(field->cs, nlocal);

  /* Copy initial values; then the kernel... */
  tdpAssert(tdpMalloc((void **) &dsum, sizeof(sum_t)));
  tdpAssert(tdpMemcpy(dsum, &sum_local, sizeof(sum_t), tdpMemcpyHostToDevice));

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(field->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(stats_field_q_kernel, nblk, ntpb, 0, 0, k3d,
		    field->target, map->target, nxx, dsum);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpDeviceSynchronize() );
  }

  tdpAssert(tdpMemcpy(&sum_local, dsum, sizeof(sum_t), tdpMemcpyDeviceToHost));

  MPI_Reduce(&sum_local.qmin, &sum->qmin, 1, MPI_DOUBLE, MPI_MIN, rank, comm);
  MPI_Reduce(&sum_local.qmax, &sum->qmax, 1, MPI_DOUBLE, MPI_MAX, rank, comm);
  MPI_Reduce(&sum_local.qvar, &sum->qvar, 1, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(&sum_local.vol,  &sum->vol,  1, MPI_DOUBLE, MPI_SUM, rank, comm);

  {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Op op = MPI_OP_NULL;

    klein_mpi_datatype(&dt);
    klein_mpi_op_sum(&op);

    MPI_Reduce(&sum_local.qsum, &sum->qsum, 1, dt, op, rank, comm);

    MPI_Op_free(&op);
    MPI_Type_free(&dt);
  }

  tdpFree(dsum);

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_q_kernel
 *
 *  Kernel for one order parameter entry "nxx",
 *
 *****************************************************************************/

__global__ void stats_field_q_kernel(kernel_3d_t k3d, field_t * field,
				     map_t * map, int nxx, sum_t * sum) {
  int kindex = 0;
  int tid = threadIdx.x;
  __shared__ sum_t lsum[TARGET_MAX_THREADS_PER_BLOCK];

  assert(field);
  assert(map);

  /* Local sum */

  lsum[tid] = sum_zero();

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double q0[NQAB] = {0};

      field_scalar_array(field, index, q0);

      klein_add_double(&lsum[tid].qsum, q0[nxx]);
      lsum[tid].qvar += q0[nxx]*q0[nxx];
      lsum[tid].qmin  = double_min(lsum[tid].qmin, q0[nxx]);
      lsum[tid].qmax  = double_max(lsum[tid].qmax, q0[nxx]);
      lsum[tid].vol  += 1.0;
    }
  }

  __syncthreads();

  if (tid == 0) {

    /* Accumulate each total for this block */

    sum_t bsum = sum_zero();

    for (int it = 0; it < blockDim.x; it++) {
      klein_add(&bsum.qsum, lsum[it].qsum);
      bsum.qvar += lsum[it].qvar;
      bsum.qmin  = double_min(bsum.qmin, lsum[it].qmin);
      bsum.qmax  = double_max(bsum.qmax, lsum[it].qmax);
      bsum.vol  += lsum[it].vol;
    }

    /* Accumulate to final result with protected update */

    while (atomicCAS(&sum->qsum.lock, 0, 1) != 0)
      ;
    __threadfence();

    klein_add(&sum->qsum, bsum.qsum);

    __threadfence();
    atomicExch(&sum->qsum.lock, 0);

    tdpAtomicAddDouble(&sum->qvar, bsum.qvar);
    tdpAtomicMinDouble(&sum->qmin, bsum.qmin);
    tdpAtomicMaxDouble(&sum->qmax, bsum.qmax);
    tdpAtomicAddDouble(&sum->vol,  bsum.vol);
  }

  return;
}

/*****************************************************************************
 *
 *  stats_field_info_bbl
 *
 *  With correction for bbl for conserved order parameters (largely
 *  for binary fluid).
 *
 *****************************************************************************/

int stats_field_info_bbl(field_t * obj, map_t * map, bbl_t * bbl) {

  int n, nf;
  MPI_Comm comm;

  double fmin[NQAB];
  double fmax[NQAB];
  double fsum[NQAB];
  double fvar[NQAB];
  double fbbl[NQAB];
  double fbbl_local[NQAB];
  double fvol, rvol;
  double fbar, f2;

  assert(obj);
  assert(map);
  assert(bbl);

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  pe_mpi_comm(obj->pe, &comm);
  stats_field_reduce(obj, map, fmin, fmax, fsum, fvar, &fvol, 0, comm);

  /* BBL corrections to be added */
  for (n = 0; n < nf; n++) {
    fbbl_local[n] = 0.0;
  }
  bbl_order_parameter_deficit(bbl, fbbl_local);
  MPI_Reduce(fbbl_local, fbbl, nf, MPI_DOUBLE, MPI_SUM, 0, comm);
  for (n = 0; n < nf; n++) {
    fsum[n] += fbbl[n];
  }

  rvol = 1.0 / fvol;

  for (n = 0; n < nf; n++) {

    fbar = rvol*fsum[n];                 /* mean */
    f2   = rvol*fvar[n]  - fbar*fbar;    /* variance */

    pe_info(obj->pe, "[phi] %14.7e %14.7e%14.7e %14.7e%14.7e\n",
	    fsum[n], fbar, f2, fmin[n], fmax[n]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_reduce
 *
 *  This is a global reduction to rank in communicator comm.
 *
 *  We expect and assert NQAB to be the largest number of field elements
 *  to avoid memory allocation and deallocation here.
 *
 *****************************************************************************/

int stats_field_reduce(field_t * obj, map_t * map, double * fmin,
		       double * fmax,  double * fsum, double * fvar,
		       double * fvol, int rank, MPI_Comm comm) {
  int nf;

  double fmin_local[NQAB];
  double fmax_local[NQAB];
  double fsum_local[NQAB];
  double fvar_local[NQAB];
  double fvol_local[1];

  assert(obj);
  assert(map);

  field_nf(obj, &nf);
  assert(nf <= NQAB);

  stats_field_local(obj, map, fmin_local, fmax_local, fsum_local, fvar_local,
		    fvol_local);

  MPI_Reduce(fmin_local, fmin, nf, MPI_DOUBLE, MPI_MIN, rank, comm);
  MPI_Reduce(fmax_local, fmax, nf, MPI_DOUBLE, MPI_MAX, rank, comm);
  MPI_Reduce(fsum_local, fsum, nf, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(fvar_local, fvar, nf, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(fvol_local, fvol,  1, MPI_DOUBLE, MPI_SUM, rank, comm);

  return 0;
}

/*****************************************************************************
 *
 *  stats_field_local
 *
 *  Accumulate the local statistics for each field scalar:
 *
 *   fmin[]  minimum
 *   fmax[]  maximum
 *   fsum[]  the sum
 *   fvar[]  the sum of the squares used to compute the variance
 *   fvol    volume of fluid required to get the mean
 *
 *   Each of the arrays must be large enough to hold the value for
 *   a field with nf elements.
 *
 *****************************************************************************/

int stats_field_local(field_t * obj, map_t * map, double * fmin, double * fmax,
		      double * fsum, double * fvar, double * fvol) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nf;
  int status;

  double f0[NQAB];

  assert(obj);
  assert(fmin);
  assert(fmax);
  assert(fsum);
  assert(map);

  cs_nlocal(obj->cs, nlocal);
  field_nf(obj, &nf);
  assert(nf <= NQAB);

  *fvol = 0.0;

  for (n = 0; n < nf; n++) {
    fmin[n] = +DBL_MAX;
    fmax[n] = -DBL_MAX;
    fsum[n] = 0.0;
    fvar[n] = 0.0;
  }

  /* Local sum */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(obj->cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	*fvol += 1.0;
	field_scalar_array(obj, index, f0);

	for (n = 0; n < nf; n++) {
	  fmin[n] = double_min(fmin[n], f0[n]);
	  fmax[n] = double_max(fmax[n], f0[n]);
	  fsum[n] += f0[n];
	  fvar[n] += f0[n]*f0[n];
	}

      }
    }
  }

  return 0;
}
