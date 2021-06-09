/*****************************************************************************
 *
 *  util_sum.c
 *
 *  Compensated sums. File to be compiled at -O0.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "util_sum.h"

/*****************************************************************************
 *
 *  kahan_add
 *
 *  Add a contribution to a compensated sum.
 *
 *****************************************************************************/

__host__ __device__ void kahan_add(kahan_t * kahan, double val) {

  assert(kahan);

  {
    volatile double y = val - kahan->cs;
    volatile double t = kahan->sum + y;
    kahan->cs  = (t - kahan->sum) - y;
    kahan->sum = t;
  }
}

/*****************************************************************************
 *
 *  klein_zero
 *
 *  Return zero object.
 *
 *****************************************************************************/

__host__ __device__ klein_t klein_zero(void) {

  klein_t sum = {0, 0.0, 0.0, 0.0};

  return sum;
}

/*****************************************************************************
 *
 *  klein_add
 *
 *  Add a contribution to a doubly compensated sum.
 *
 *****************************************************************************/

__host__ __device__ void klein_add(klein_t * klein, double val) {

  assert(klein);

  {
    volatile double c;
    volatile double cc;
    volatile double t = klein->sum + val;
  
    if (fabs(klein->sum) >= fabs(val)) {
      c = (klein->sum - t) + val;
    }
    else {
      c = (val - t) + klein->sum;
    }
    klein->sum = t;
    t = klein->cs + c;
    if (fabs(klein->cs) >= fabs(c)) {
      cc = (klein->cs - t) + c;
    }
    else {
      cc = (c - t) + klein->cs;
    }
    klein->cs = t;
    klein->ccs = klein->ccs + cc;
  }

  return;
}

/*****************************************************************************
 *
 *  klein_sum
 *
 *  Convenience to return doubly compensated sum.
 *
 *****************************************************************************/

__host__ __device__ double klein_sum(const klein_t * klein) {

  assert(klein);

  return klein->sum + klein->cs + klein->ccs;
}

/*****************************************************************************
 *
 *  klein_atomic_add
 *
 *  Add contribution atomically
 *
 *****************************************************************************/

__host__ __device__ void klein_atomic_add(klein_t * sum, double val) {

  assert(sum);

#ifdef __CUDA_ARCH___
  while (atomicCAS(&sum->lock, 0, 1) != 0) {};
  __threadfence();
#endif

  klein_add(sum, val);

#ifdef __CUDA_ARCH__
  __threadfence();
  atomicExch(&sum->lock, 0);
#endif

  return;
}

/*****************************************************************************
 *
 *  klein_mpi_datatype
 *
 *  Generate a datatype handle. Caller to release via MPI_Datatype_free().
 *
 *****************************************************************************/

__host__ int klein_mpi_datatype(MPI_Datatype * type) {

  int ierr = 0;
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Aint displacements[5] = {};
  MPI_Datatype types[4] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}; 

  klein_t sum = {};

  assert(type);

  MPI_Get_address(&sum,      displacements + 0);
  MPI_Get_address(&sum.lock, displacements + 1);
  MPI_Get_address(&sum.sum,  displacements + 2);
  MPI_Get_address(&sum.cs,   displacements + 3);
  MPI_Get_address(&sum.ccs,  displacements + 4);

  /* Subtract the offset (displacements[0] == displacements[1] in fact) */
  for (int n = 1; n <= 4; n++) {
    displacements[n] -= displacements[0];
  }

  MPI_Type_create_struct(4, blocklengths, displacements + 1, types, type);
  MPI_Type_commit(type);

  return ierr;
}

/*****************************************************************************
 *
 *  klein_mpi_op_sum_function
 *
 *  Implementation for sum operation below.
 *
 *****************************************************************************/

__host__ void klein_mpi_op_sum_function(klein_t * invec, klein_t * inoutvec,
					int * len, MPI_Datatype * dt) {

  assert(invec);
  assert(inoutvec);
  assert(len);
  assert(dt);

  for (int n = 0; n < *len; n++) {
    klein_add(inoutvec + n, invec[n].cs);
    klein_add(inoutvec + n, invec[n].ccs);
    klein_add(inoutvec + n, invec[n].sum);
  }

  return;
}

/*****************************************************************************
 *
 *  klein_mpi_op_sum
 *
 *  Return the MPI_Op for the sum above. Caller to MPI_Op_free().
 *
 *****************************************************************************/

__host__ int klein_mpi_op_sum(MPI_Op * op) {

  assert(klein_mpi_op_sum_function);
  assert(op);

  MPI_Op_create((MPI_User_function *) klein_mpi_op_sum_function, 0, op);

  return 0;
}
