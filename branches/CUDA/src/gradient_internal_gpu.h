/*****************************************************************************
 *
 *  gradient_internal_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _GRADIENT_INRERNAL_GPU_H
#define _GRADIENT_INTERNAL_GPU_H

#include "common_gpu.h"
#define NOP 5

/* supported schemes for GPU version */
enum gradent_options_gpu {OPTION_3D_7PT_FLUID,OPTION_3D_7PT_SOLID};

/* required external routines */
extern "C" double blue_phase_q0(void);
extern "C" double blue_phase_kappa0(void);
extern "C" double blue_phase_kappa1(void);
extern "C" double colloids_q_tensor_w(void);
extern "C" void coords_nlocal_offset(int n[3]);
extern "C" double blue_phase_amplitude_compute(void);

extern "C" void checkCUDAError(const char *msg);


/* forward declarations  */
void set_gradient_option_gpu(char option);
void put_gradient_constants_on_gpu();

__global__ void gradient_3d_7pt_fluid_operator_gpu_d(int nop, int nhalo, 
						     int N[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     int * le_index_real_to_buffer_d,
						     int nextra);

__global__ void gradient_3d_7pt_solid_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     char * site_map_status_d,
					    char * colloid_map_d,
					    double * colloid_r_d,
					    int nextra);

__device__ static void gradient_bcs_gpu_d(const double kappa0, 
					  const double kappa1, 
					  const int dn[3],
					  double dq[NOP][3], 
					  double bc[NOP][NOP][3]);
__device__ static int util_gaussian_gpu_d(double a[NOP][NOP], double xb[NOP]);
__device__ void colloids_q_boundary_normal_gpu_d(const int di[3],
						 double dn[3], 
						 int Nall[3], 
						 int nhalo, int nextra, 
						 int ii, int jj, int kk, 
						 char *site_map_status_d,
						 char * colloid_map_d,
						 double * colloid_r_d);




__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);

/* constant memory symbols internal to this module */
__constant__ double q_0_cd;
__constant__ double kappa0_cd;
__constant__ double kappa1_cd;
__constant__ double kappa2_cd;
__constant__ double w_cd;
__constant__ double amplitude_cd;
__constant__ double e_cd[3][3][3];
__constant__ int noffset_cd[3];
__constant__ double d_cd[3][3];
__constant__ char bcs_cd[6][3];


/* external variables holding device memory addresses */
extern double * phi_site_d;
extern double * grad_phi_site_d;
extern double * delsq_phi_site_d;
extern int * N_d;
extern int * le_index_real_to_buffer_d;
extern char * site_map_status_d;
extern char * colloid_map_d;
extern double * colloid_r_d;

#endif
