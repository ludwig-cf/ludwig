/*****************************************************************************
 *
 *  collision_internal_gpu.h
 *
 *  Alan Gray
 * 
 *****************************************************************************/

#ifndef COLLISION_INTERNAL_GPU_H
#define COLLISION_INTERNAL_GPU_H

#include "common_gpu.h"

enum lattchunks {ALL,BULK,EDGES};
enum colltype {MULTIRELAXATION,BINARY};


/* Declarations for gpu kernel/device routines  */
__global__ void collision_multirelaxation_gpu_d(int ndist, int nhalo, 
						int N[3],	      
						const double* __restrict__ force_global_d, 
						double* __restrict__ f_d,		
						const double* __restrict__ ftmp_d,		
						const char* __restrict__ site_map_status_d, 
					      const double* __restrict__ force_ptr, 
						double* __restrict__ velocity_ptr);

__global__ void collision_lb_gpu_d(int ndist, int nhalo, int N[3], 
					  const double* __restrict__ force_global_d, 
					  double* __restrict__ f_d,
				   const double* __restrict__ ftmp_d,					
					  const char* __restrict__ site_map_status_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_ptr, 
					  double* __restrict__ velocity_ptr, 
					  int colltype, int latchunk);

__global__ static void collision_edge_gpu_d(int nhalo, 
						   int N[3],
						   const double* __restrict__ force_global_d, 
						   double* __restrict__ f_d, 
					    const double* __restrict__ ftmp_d,		
						   const char* __restrict__ site_map_status_d, 
						   const double* __restrict__ phi_site_d,		
						   const double* __restrict__ grad_phi_site_d,	
						   const double* __restrict__ delsq_phi_site_d,	
						   const double* __restrict__ force_d, 
					    double* __restrict__ velocity_d,int colltype, int dirn);


__device__ void fluctuations_off_gpu_d(double shat[3][3], double ghat[NVEL]);
__device__ double symmetric_chemical_potential_gpu_d(const int index,	
						     const double* __restrict__ phi_site_d,
						     const double* __restrict__ delsq_phi_site_d);
__device__ void symmetric_chemical_stress_gpu_d(const int index, 
						double s[3][3],
						const double* __restrict__ phi_site_d, 
						const double* __restrict__ grad_phi_site_d, 
						const double* __restrict__ delsq_phi_site_d,
						int nsite);
__device__ double dot_product_gpu_d(const double a[3], const double b[3]);
__device__ double phi_get_delsq_delsq_phi_site_gpu_d(const int index,	
					  double *delsq_delsq_phi_site_d);
__device__ double phi_get_delsq_phi_site_gpu_d(const int index, 
					       double *delsq_phi_site_d);
__device__ void phi_get_grad_phi_site_gpu_d(const int index, double grad[3], 
					    double *grad_phi_site_d);
__device__ double phi_get_phi_site_gpu_d(const int index, double *phi_site_d);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3]);

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

/* external variables holding GPU memory addresses */
extern double * f_d;
extern double * ftmp_d;
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
__constant__ double rtau_d[NVEL];
__constant__ double wv_cd[NVEL];
__constant__ double ma_cd[NVEL][NVEL];
__constant__ double mi_cd[NVEL][NVEL];
__constant__ double q_cd[NVEL][3][3];
__constant__ int cv_cd[NVEL][3];
__constant__ double d_cd[3][3];
__constant__ double a_d;
__constant__ double b_d;
__constant__ double kappa_d;
__constant__ double rtau2_d;
__constant__ double rcs2_d;



#endif
