/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _GRADIENT_3D_7PT_FLUID_GPU_H
#define _GRADIENT_3D_7PT_FLUID_GPU_H

#include "common_gpu.h"

/* expose main routine in this module to outside routines */
extern "C" void phi_gradients_compute_gpu(void);

/* forward declarations of device routines */
__global__ void gradient_3d_7pt_fluid_operator_gpu_d(int nop, int nhalo, 
						     int N[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     int * le_index_real_to_buffer_d,
						     int nextra);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);

/* external variables holding device memory addresses */
extern double * phi_site_d;
extern double * grad_phi_site_d;
extern double * delsq_phi_site_d;
extern int * N_d;
extern int * le_index_real_to_buffer_d;

#endif
