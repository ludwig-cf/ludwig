/*****************************************************************************
 *
 *  field_stats_q.c
 *
 *  Order parameter statistics for Q_ab tansor order parameters.
 *
 *  The single kernel here could be split a number of ways: one
 *  kernel per scalar order parameter, or one kernel per global
 *  quantity; or some combination thereof.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "field_stats_q.h"
#include "kernel.h"
#include "util_sum.h"

typedef struct sum_s sum_t;
struct sum_s {
  klein_t qsum;             /* Sensitive to round-off */
  double  qvar;             /* Computed in single-sweep form */
  double  qmin;             /* minimum */
  double  qmax;             /* maximum */
  double  vol;              /* Volume is the same in all cases */
};

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

int field_stats_q_reduce(field_t * q, map_t * map, int nxx, sum_t * sum,
			 int rank, MPI_Comm comm);
__global__ void field_stats_q_kernel(kernel_ctxt_t * ktx, field_t * q,
				     map_t * map, int nxx, sum_t * sum);

static double double_min(double x, double y) {return (x < y) ? x : y;}
static double double_max(double x, double y) {return (x > y) ? x : y;}

/*****************************************************************************
 *
 *  field_stats_q
 *
 *****************************************************************************/

int field_stats_q(field_t * obj, map_t * map) {

  MPI_Comm comm;
  /*
  const char * qxx[5] = {"Qxx", "Qxy", "Qxz", "Qyy", "Qyz"};
  */
  const char * qxx[5] = {"phi", "phi", "phi", "phi", "phi"};

  assert(obj);
  assert(map);

  pe_mpi_comm(obj->pe, &comm);

  for (int n = 0; n < NQAB; n++) {

    sum_t sum = {0};
    field_stats_q_reduce(obj, map, n, &sum, 0, comm);

    {
      double qsum  = klein_sum(&sum.qsum);
      double rvol = 1.0/sum.vol;
      double qbar = rvol*qsum;                      /* mean */
      double qvar = rvol*sum.qvar  - qbar*qbar;     /* variance */

      pe_info(obj->pe, "[%3s] %14.7e %14.7e %14.7e %14.7e %14.7e\n",
	      qxx[n], qsum, qbar, qvar, sum.qmin, sum.qmax);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_stats_q_reduce
 *
 *  This is a global reduction to rank in communicator comm.
 *
 *  We expect and assert NQAB to be the largest number of field elements
 *  to avoid memory allocation and deallocation here.
 *
 *****************************************************************************/

int field_stats_q_reduce(field_t * obj, map_t * map, int nxx, sum_t * sum,
			 int rank, MPI_Comm comm) {

  int nlocal[3] = {0};
  sum_t sum_local = sum_zero();  /* host copy */
  sum_t * dsum = NULL;           /* device copy */

  assert(obj);
  assert(map);

  cs_nlocal(obj->cs, nlocal);

  /* Copy initial values; then the kernel... */
  tdpAssert(tdpMalloc((void **) &dsum, sizeof(sum_t)));
  tdpAssert(tdpMemcpy(dsum, &sum_local, sizeof(sum_t), tdpMemcpyHostToDevice));

  {
    kernel_info_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;
    dim3 nblk, ntpb;

    kernel_ctxt_create(obj->cs, 1, lim, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(field_stats_q_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, obj->target, map->target, nxx, dsum);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
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
 *  field_stats_q_kernel
 *
 *  Kernel for one order parameter entry "nxx",
 *
 *****************************************************************************/

void field_stats_q_kernel(kernel_ctxt_t * ktx, field_t * obj, map_t * map,
			  int nxx, sum_t * sum) {
  int kindex = 0;
  int kiterations = 0;

  assert(obj);
  assert(map);

  __shared__ sum_t lsum[TARGET_MAX_THREADS_PER_BLOCK];
  int tid = threadIdx.x;

  /* Local sum */

  lsum[tid] = sum_zero();
  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);
    int status = MAP_BOUNDARY;

    map_status(map, index, &status);

    if (status == MAP_FLUID) {
      double q0[NQAB] = {0};

      field_scalar_array(obj, index, q0);

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
