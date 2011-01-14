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

/* from coords.h */
enum cartesian_directions {X, Y, Z};


/* declarations for required external (host) routines */
extern "C" void coords_nlocal(int n[3]);
extern "C" int coords_nhalo(void);
extern "C" int coords_index(int,int,int);
extern "C" int distribution_ndist(void);
extern "C" void hydrodynamics_get_force_local(const int, double *);
extern "C" void hydrodynamics_get_velocity(const int, double *);
extern "C" void hydrodynamics_set_velocity(const int, double *);
extern "C" void fluid_body_force(double f[3]);
extern "C" char site_map_get_status(int,int,int);
extern "C" double phi_get_phi_site(const int);
extern "C" void phi_set_phi_site(const int, const double);
extern "C" void   phi_gradients_grad(const int index, double grad[3]);
extern "C" double phi_gradients_delsq(const int index);
extern "C" void TIMER_start(const int);
extern "C" void TIMER_stop(const int);

/* expose routines in this module to outside routines */
extern "C" void initialise_gpu();
extern "C" void put_f_on_gpu();
extern "C" void put_force_on_gpu();
extern "C" void put_phi_on_gpu();
extern "C" void get_f_from_gpu();
extern "C" void get_velocity_from_gpu();
extern "C" void get_phi_site_from_gpu();
extern "C" void finalise_gpu();
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void checkCUDAError(const char *msg);


/* forward declarations of host routines internal to this module */
static void calculate_data_sizes(void);
static void allocate_memory_on_gpu(void);
static void free_memory_on_gpu(void);
static void   get_packed_index_offset(int *packed_index,int *offset,
				      int ii,int jj,int kk,int nhalo,
				      int N[3],int ndist);
static int get_linear_index(int ii,int jj,int kk,int N[3]);


/* forward declarations of accelerator routines internal to this module */
__global__ static void pack_edges_gpu_d(int ndist, int nhalo,
						   int maxN, int nlocal[3], 
						   double* fedge_d, 
						   double* f_d);

__global__ static void unpack_halos_gpu_d(int ndist, int nhalo, int maxN, 
					  int N[3], double* f_d, 
					  double* fhalo_d);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
 __device__ static void   get_packed_index_offset_gpu_d(int *packed_index,
							int *offset,int ii,
							int jj,int kk,
							int nhalo,
							int N[3],int ndist);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);


#endif
