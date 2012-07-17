/*****************************************************************************
 *
 *  collision_gpu.h
 *
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *  Adaptations for running kernel on GPU: Alan Gray/Alan Richardson
 * 
 *****************************************************************************/

#ifndef COLLISION_GPU_H
#define COLLISION_GPU_H

#include "common_gpu.h"

/* Declarations for gpu kernel/device routines  */
__global__ void collision_multirelaxation_gpu_d(int ndist, int nhalo, 
						int N[3],	      
						double* force_global_d, 
						double* f_d,		
						char* site_map_status_d, 
					      double* force_ptr, 
						double* velocity_ptr,	
						double* ma_ptr,		
						double* d_ptr,		
						double* mi_ptr);

__global__ void collision_binary_lb_gpu_d(int ndist, int nhalo, int N[3], 
					  double* force_global_d, 
					  double* f_d,			
					  char* site_map_status_d, 
					  double* phi_site_d,		
					  double* grad_phi_site_d,	
					  double* delsq_phi_site_d,	
					  double* force_ptr, 
					  double* velocity_ptr, 
					  double* ma_ptr, 
					  double* d_ptr, 
					  double* mi_ptr, 
					  int* cv_ptr, 
					  double* q_ptr, 
					  double* wv_d);

__device__ void fluctuations_off_gpu_d(double shat[3][3], double ghat[NVEL]);
__device__ double symmetric_chemical_potential_gpu_d(const int index,	
						     double *phi_site_d,
						     double *delsq_phi_site_d);
__device__ void symmetric_chemical_stress_gpu_d(const int index, 
						double s[3][3],
						double *phi_site_d, 
						double *grad_phi_site_d, 
						double *delsq_phi_site_d,
						double d_d[3][3]);
__device__ double dot_product_gpu_d(const double a[3], const double b[3]);
__device__ double phi_get_delsq_delsq_phi_site_gpu_d(const int index,	
					  double *delsq_delsq_phi_site_d);
__device__ double phi_get_delsq_phi_site_gpu_d(const int index, 
					       double *delsq_phi_site_d);
__device__ void phi_get_grad_phi_site_gpu_d(const int index, double grad[3], 
					    double *grad_phi_site_d);
__device__ double phi_get_phi_site_gpu_d(const int index, double *phi_site_d);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);

/* Declarations for host routines */
void collision_relaxation_times_set_gpu(void);
void copy_constants_to_gpu(void);

/* declarations for required external (host) routines */
extern "C" void collision_relaxation_times_set(void);
extern "C" double get_eta_shear(void);
extern "C" double get_eta_bulk(void);
extern "C" double fluid_kt(void);
extern "C" double phi_cahn_hilliard_mobility(void);
extern "C" int  RUN_get_double_parameter(const char *, double *);

/* expose main routine in this module to outside routines */
extern "C" void collide_gpu();

/* external variables holding GPU memory addresses */
extern double * f_d;
extern double * ma_d;
extern double * mi_d;
extern double * d_d;
extern int * cv_d;
extern double * q_d;
extern double * wv_d;
extern char * site_map_status_d;
extern double * force_temp;
extern double * velocity_temp;
extern double * force_d;
extern double * velocity_d;
extern int * N_d;
extern double * force_global_d;
extern double * phi_site_d;
extern double * grad_phi_site_d;
extern double * delsq_phi_site_d;

/* constant variables on accelerator (on-chip read-only memory) */
__constant__ double rtau_shear_d;
__constant__ double rtau_bulk_d;
__constant__ double rtau_ghost_d;
__constant__ double a_d;
__constant__ double b_d;
__constant__ double kappa_d;
__constant__ double rtau2_d;
__constant__ double rcs2_d;



#endif
