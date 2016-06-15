/*****************************************************************************
 *
 *  kernel.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_KERNEL_H
#define LUDWIG_KERNEL_H

#include "targetDP.h"

typedef struct kernel_ctxt_s kernel_ctxt_t;
typedef struct kernel_info_s kernel_info_t;

/* kernel_info_t
 * is just a convenience to allow the user to pass the
 * relevant information to the context constructor. */

struct kernel_info_s {
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
};

__host__ int kernel_ctxt_create(int nsimdvl, kernel_info_t info, kernel_ctxt_t ** p);
__host__ int kernel_ctxt_launch_param(kernel_ctxt_t * obj, dim3 * nblk, dim3 * ntpb);
__host__ int kernel_ctxt_info(kernel_info_t * lim);
__host__ int kernel_ctxt_free(kernel_ctxt_t * obj);

__host__ __target__ int kernel_coords_ic(int kindex);
__host__ __target__ int kernel_coords_jc(int kindex);
__host__ __target__ int kernel_coords_kc(int kindex);
__host__ __target__ int kernel_coords_icv(int kindex, int iv);
__host__ __target__ int kernel_coords_jcv(int kindex, int iv);
__host__ __target__ int kernel_coords_kcv(int kindex, int iv);
__host__ __target__ int kernel_mask(int ic, int jc, int kc);
__host__ __target__ int kernel_coords_index(int ic, int jc, int kc);
__host__ __target__ int kernel_iterations(void);
__host__ __target__ int kernel_vector_iterations(void);

__host__ int kernel_launch_param(int iterations, dim3 * nblk, dim3 * ntpb);

#endif

