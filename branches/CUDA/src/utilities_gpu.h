/*****************************************************************************
 * 
 * utilities_gpu.h
 * 
 * Data management and other utilities for GPU adaptation of Ludwig
 * Alan Gray/ Alan Richardson 
 *
 *****************************************************************************/

#ifndef UTILITIES_GPU_H
#define UTILITIES_GPU_H

#include "common_gpu.h"

/* declarations for required external (host) routines */
extern "C" void hydrodynamics_get_force_local(const int, double *);
extern "C" void hydrodynamics_set_force_local(const int, double *);
extern "C" void hydrodynamics_get_velocity(const int, double *);
extern "C" void hydrodynamics_set_velocity(const int, double *);
extern "C" void fluid_body_force(double f[3]);
extern "C" char site_map_get_status(int,int,int);


/* expose routines in this module to outside routines */
extern "C" void initialise_gpu();
extern "C" void put_site_map_on_gpu();
extern "C" void put_force_on_gpu();
extern "C" void put_velocity_on_gpu();
extern "C" void get_force_from_gpu();
extern "C" void zero_force_on_gpu();
extern "C" void get_velocity_from_gpu();
extern "C" void finalise_gpu();
extern "C" void checkCUDAError(const char *msg);


/* forward declarations of host routines internal to this module */
static void calculate_data_sizes(void);
static void allocate_memory_on_gpu(void);
static void free_memory_on_gpu(void);
int get_linear_index(int ii,int jj,int kk,int N[3]);


/* forward declarations of accelerator routines internal to this module */

__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);


#endif
