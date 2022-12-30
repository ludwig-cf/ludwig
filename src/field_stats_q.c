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
  klein_t qsum[NQAB];     /* Avoid summation order differences */
  double  qvar[NQAB];
  double  qmin[NQAB];
  double  qmax[NQAB];
  double  vol;
};

int field_stats_q_reduce(field_t * q, map_t * map, sum_t * sum, int, MPI_Comm);
__global__ void field_stats_q_kernel(kernel_ctxt_t * ktx, field_t * q,
				     map_t * map, sum_t * sum);

static double double_min(double x, double y) {return (x < y) ? x : y;}
static double double_max(double x, double y) {return (x > y) ? x : y;}

/*****************************************************************************
 *
 *  field_stats_q
 *
 *****************************************************************************/

int field_stats_q(field_t * obj, map_t * map) {

  MPI_Comm comm;
  sum_t sum = {0};
  /*
  const char * qxx[5] = {"Qxx", "Qxy", "Qxz", "Qyy", "Qyz"};
  */
  const char * qxx[5] = {"phi", "phi", "phi", "phi", "phi"};
  assert(obj);
  assert(map);

  pe_mpi_comm(obj->pe, &comm);
  field_stats_q_reduce(obj, map, &sum, 0, comm);

  for (int n = 0; n < NQAB; n++) {

    double qsum  = klein_sum(sum.qsum + n);
    double rvol = 1.0/sum.vol;
    double qbar = rvol*qsum;                         /* mean */
    double qvar = rvol*sum.qvar[n]  - qbar*qbar;     /* variance */

    pe_info(obj->pe, "[%3s] %14.7e %14.7e %14.7e %14.7e %14.7e\n",
	    qxx[n], qsum, qbar, qvar, sum.qmin[n], sum.qmax[n]);
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

int field_stats_q_reduce(field_t * obj, map_t * map, sum_t * sum,
			 int rank, MPI_Comm comm) {

  int nlocal[3] = {0};
  sum_t sum_local = {0};

  assert(obj);
  assert(map);

  /* Local sum */
  for (int n = 0; n < NQAB; n++) {
    sum_local.qsum[n] = klein_zero();
    sum_local.qvar[n] = 0.0;
    sum_local.qmin[n] = +DBL_MAX;
    sum_local.qmax[n] = -DBL_MAX;
  }
  sum_local.vol = 0.0;

  cs_nlocal(obj->cs, nlocal);

  {
    kernel_info_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;
    dim3 nblk, ntpb;

    kernel_ctxt_create(obj->cs, 1, lim, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    /* FIXME: sum_local device memory */
    tdpLaunchKernel(field_stats_q_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, obj->target, map->target, &sum_local);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  MPI_Reduce(sum_local.qmin, sum->qmin, NQAB, MPI_DOUBLE, MPI_MIN, rank, comm);
  MPI_Reduce(sum_local.qmax, sum->qmax, NQAB, MPI_DOUBLE, MPI_MAX, rank, comm);
  MPI_Reduce(sum_local.qvar, sum->qvar, NQAB, MPI_DOUBLE, MPI_SUM, rank, comm);
  MPI_Reduce(&sum_local.vol, &sum->vol,    1, MPI_DOUBLE, MPI_SUM, rank, comm);

  {
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Op op = MPI_OP_NULL;

    klein_mpi_datatype(&dt);
    klein_mpi_op_sum(&op);

    MPI_Reduce(sum_local.qsum, sum->qsum, NQAB, dt, op, rank, comm);

    MPI_Op_free(&op);
    MPI_Type_free(&dt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_stats_q_kernel
 *
 *****************************************************************************/

void field_stats_q_kernel(kernel_ctxt_t * ktx, field_t * obj, map_t * map,
			  sum_t * sum) {
  int kindex = 0;
  int kiterations = 0;

  assert(obj);
  assert(map);

  __shared__ sum_t lsum[TARGET_MAX_THREADS_PER_BLOCK];
  int tid = threadIdx.x;

  /* Local sum */

  for (int n = 0; n < NQAB; n++) {
    lsum[tid].qsum[n] = klein_zero();
    lsum[tid].qvar[n] = 0.0;
    lsum[tid].qmin[n] = +DBL_MAX;
    lsum[tid].qmax[n] = -DBL_MAX;
  }
  lsum[tid].vol = 0.0;

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

      lsum[tid].vol += 1.0;
      field_scalar_array(obj, index, q0);

      for (int n = 0; n < NQAB; n++) {
	lsum[tid].qmin[n] = double_min(lsum[tid].qmin[n], q0[n]);
	lsum[tid].qmax[n] = double_max(lsum[tid].qmax[n], q0[n]);
	lsum[tid].qvar[n] += q0[n]*q0[n];
	klein_add_double(&lsum[tid].qsum[n], q0[n]);
      }
    }
  }

  __syncthreads();

  if (tid == 0) {

    /* Accumulate each total for this block */

    sum_t bsum = {0};

    for (int n = 0; n < NQAB; n++) {
      bsum.qsum[n] = klein_zero();
      bsum.qvar[n] = 0.0;
      bsum.qmin[n] = +DBL_MAX;
      bsum.qmax[n] = -DBL_MAX;
      bsum.vol     = 0.0;
    }

    for (int it = 0; it < blockDim.x; it++) {
      for (int n = 0; n < NQAB; n++) {
	klein_add(&bsum.qsum[n], lsum[it].qsum[n]);
	bsum.qvar[n] += lsum[it].qvar[n];
	bsum.qmin[n]  = double_min(bsum.qmin[n], lsum[it].qmin[n]);
	bsum.qmax[n]  = double_max(bsum.qmax[n], lsum[it].qmax[n]);
      }
      bsum.vol += lsum[it].vol;
    }

    /* Accumulate to final result */
    for (int n = 0; n < NQAB; n++) {
      /* FIXME: start unsafe update */
      klein_add(&sum->qsum[n], bsum.qsum[n]);
      /* end */
      tdpAtomicAddDouble(&sum->qvar[n], bsum.qvar[n]);
      tdpAtomicMinDouble(&sum->qmin[n], bsum.qmin[n]);
      tdpAtomicMaxDouble(&sum->qmax[n], bsum.qmax[n]);
    }
    tdpAtomicAddDouble(&sum->vol, bsum.vol);
  }

  return;
}
