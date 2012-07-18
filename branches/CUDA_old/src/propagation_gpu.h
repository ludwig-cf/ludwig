/*****************************************************************************
 *
 *  propagation_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PROPAGATION_GPU_H
#define _PROPAGATION_GPU_H

#include "common_gpu.h"
#include "dist_datamgmt_gpu.h"

/* expose main routine in this module to outside routines */
extern "C" void propagation_gpu();

/* forward declarations of device routines */
__global__ static void propagate_d3q19_gpu_d(int ndist, int nhalo, int N[3], \
					     double* fnew_d, double* fold_d);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3]);

/* external variables holding device memory addresses */
extern double * f_d;
extern double * ftmp_d;
extern int * N_d;

#endif
