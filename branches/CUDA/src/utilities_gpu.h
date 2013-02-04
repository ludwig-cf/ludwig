/*****************************************************************************
 * 
 * utilities_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef UTILITIES_GPU_H
#define UTILITIES_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose routines in this module to outside routines */
CFUNC void initialise_gpu();
CFUNC void put_site_map_on_gpu();
CFUNC void put_colloid_map_on_gpu();
CFUNC void put_colloid_properties_on_gpu();
CFUNC void zero_colloid_force_on_gpu();
CFUNC void get_fluxes_from_gpu();
CFUNC void put_fluxes_on_gpu();
CFUNC void finalise_gpu();
CFUNC void checkCUDAError(const char *msg);
CFUNC int get_linear_index(int ii,int jj,int kk,int N[3]);

#endif
