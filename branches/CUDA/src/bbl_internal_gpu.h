/*****************************************************************************
 * 
 * bbl_internal_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef BBL_INTERNAL_GPU_H
#define BBL_INTERNAL_GPU_H

#include "common_gpu.h"

__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);

#endif
