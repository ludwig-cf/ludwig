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

typedef struct kernel_lim_s kernel_lim_t;

struct kernel_lim_s {
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
};

__host__            int kernel_coords_commit(kernel_lim_t limits);
__host__            int kernel_lim(kernel_lim_t * lim);
__host__            int kernel_launch_param(int nvl, dim3 * nblk, dim3 * ntpb);
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

#endif
