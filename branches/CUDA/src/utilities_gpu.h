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
enum cartesian_neighbours {FORWARD, BACKWARD};

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
extern "C" int le_get_nxbuffer(void);
extern "C" int le_index_real_to_buffer(const int ic, const int di);
extern "C" int    phi_nop(void);
extern "C" MPI_Comm cart_comm(void);
extern "C" int    cart_size(const int);
extern "C" int    cart_neighb(const int direction, const int dimension);
extern "C" int    cart_rank(void);

/* expose routines in this module to outside routines */
extern "C" void initialise_gpu();
extern "C" void put_f_on_gpu();
extern "C" void put_force_on_gpu();
extern "C" void put_phi_on_gpu();
extern "C" void put_grad_phi_on_gpu();
extern "C" void put_delsq_phi_on_gpu();
extern "C" void get_f_from_gpu();
extern "C" void get_velocity_from_gpu();
extern "C" void get_phi_from_gpu();
extern "C" void finalise_gpu();
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void halo_swap_gpu(void);
extern "C" void get_phi_edges_from_gpu(void);
extern "C" void put_phi_halos_on_gpu(void);
extern "C" void phi_halo_swap_gpu(void);
extern "C" void checkCUDAError(const char *msg);


/* forward declarations of host routines internal to this module */
static void calculate_data_sizes(void);
static void allocate_memory_on_gpu(void);
static void free_memory_on_gpu(void);

static void calculate_dist_data_sizes(void);
static void allocate_dist_memory_on_gpu(void);
static void free_dist_memory_on_gpu(void);

static void calculate_phi_data_sizes(void);
static void allocate_phi_memory_on_gpu(void);
static void free_phi_memory_on_gpu(void);


void init_dist_gpu();
void finalise_dist_gpu();


void init_phi_gpu();
void finalise_phi_gpu();

int get_linear_index(int ii,int jj,int kk,int N[3]);


/* forward declarations of accelerator routines internal to this module */
__global__ static void pack_edgesX_gpu_d(int ndist, int nhalo,
					 int* cv_d, int N[3], 
					 double* fedgeXLOW_d,
					 double* fedgeXHIGH_d, double* f_d); 
__global__ static void unpack_halosX_gpu_d(int ndist, int nhalo, int N[3],
					 int* cv_d, 
					   double* f_d, double* fhaloXLOW_d,
					   double* fhaloXHIGH_d);
__global__ static void pack_edgesY_gpu_d(int ndist, int nhalo,
					 int* cv_d, int N[3], 
					 double* fedgeYLOW_d,
					 double* fedgeYHIGH_d, double* f_d); 
__global__ static void unpack_halosY_gpu_d(int ndist, int nhalo, int N[3],
					 int* cv_d, 
					   double* f_d, double* fhaloYLOW_d,
					   double* fhaloYHIGH_d);
__global__ static void pack_edgesZ_gpu_d(int ndist, int nhalo, 
					 int* cv_d, int N[3],  
					 double* fedgeZLOW_d,
					 double* fedgeZHIGH_d, double* f_d); 
__global__ static void unpack_halosZ_gpu_d(int ndist, int nhalo, 
					 int* cv_d, int N[3], 
					   double* f_d, double* fhaloZLOW_d,
					   double* fhaloZHIGH_d);

__global__ static void pack_phi_edgesX_gpu_d(int ndist, int nhalo,
					 int N[3], 
					 double* phiedgeXLOW_d,
					 double* phiedgeXHIGH_d, double* phi_d); 
__global__ static void unpack_phi_halosX_gpu_d(int ndist, int nhalo, int N[3],
					 double* phi_d, double* phihaloXLOW_d,
					   double* phihaloXHIGH_d);
__global__ static void pack_phi_edgesY_gpu_d(int ndist, int nhalo,
					 int N[3], 
					 double* phiedgeYLOW_d,
					 double* phiedgeYHIGH_d, double* phi_d); 
__global__ static void unpack_phi_halosY_gpu_d(int ndist, int nhalo, int N[3],
					   double* phi_d, double* phihaloYLOW_d,					   double* phihaloYHIGH_d);
__global__ static void pack_phi_edgesZ_gpu_d(int ndist, int nhalo, 
					 int N[3],  
					 double* phiedgeZLOW_d,
					 double* phiedgeZHIGH_d, double* phi_d); 
__global__ static void unpack_phi_halosZ_gpu_d(int ndist, int nhalo, 
					 int N[3], 
					   double* phi_d, double* phihaloZLOW_d,					   double* phihaloZHIGH_d);

__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);


#endif
