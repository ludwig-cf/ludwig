/*****************************************************************************
 *
 *  propagation_internal_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PROPAGATION_INTERNAL_GPU_H
#define _PROPAGATION_INTERNAL_GPU_H

#include "common_gpu.h"

/* forward declarations of device routines */
__global__ static void propagate_d3q19_gpu_d(int ndist, int nhalo, int N[3], \
					     double* __restrict__ fnew_d, const double* __restrict__ fold_d);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3]);

/* external variables holding device memory addresses */
extern double * f_d;
extern double * ftmp_d;
extern int * N_d;

#endif
