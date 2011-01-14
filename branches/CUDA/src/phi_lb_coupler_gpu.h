/*****************************************************************************
 *
 *  phi_lb_coupler_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PHI_LB_COUPLER_GPU_H
#define _PHI_LB_COUPLER_GPU_H

/* Declarations for gpu kernel/device routines  */
__global__ void phi_compute_phi_site_gpu_d(int ndist, int N[3], int nhalo, \
					      double* f_d, \
						char* site_map_status_d, \
				      double* phi_site_d);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);


/* declarations for required external (host) routines */
extern "C" int    distribution_ndist(void);
extern "C" void coords_nlocal(int n[3]);
extern "C" int coords_nhalo(void);
extern "C" int    phi_is_finite_difference(void);

/* expose main routine in this module to outside routines */
extern "C" void phi_compute_phi_site_gpu();

/* from coords.h */
enum cartesian_directions {X, Y, Z};

/* external variables holding device memory addresses */
extern double * f_d;
extern char * site_map_status_d;
extern int * N_d;
extern double * phi_site_d;

#endif
