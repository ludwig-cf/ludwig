/*****************************************************************************
 * 
 * dist_datamgmt_gpu.h
 * 
 * Distribution data management utilities for GPU adaptation of Ludwig
 * Alan Gray
 *
 *****************************************************************************/

#ifndef DISTUTILITIES_GPU_H
#define DISTUTILITIES_GPU_H

#include "common_gpu.h"

/* expose routines in this module to outside routines */
extern "C" void put_f_on_gpu();
extern "C" void get_f_from_gpu();
extern "C" void put_f_partial_on_gpu(int *mask, int include_neighbours);
extern "C" void get_f_partial_from_gpu(int *mask, int include_neighbours);
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void distribution_halo_gpu(void);
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void bounce_back_gpu(int *findexall, int *linktype,
				double *dfall, double *dgall,
				double *dmall, int nlink, int pass);
extern "C" void bbl_init_temp_link_arrays_gpu(int nlink);
extern "C" void bbl_finalise_temp_link_arrays_gpu();
extern "C" void bbl_enlarge_temp_link_arrays_gpu(int nlink);

/* forward declarations of host routines internal to this module */
static void calculate_dist_data_sizes(void);
static void allocate_dist_memory_on_gpu(void);
static void free_dist_memory_on_gpu(void);
void init_dist_gpu();
void finalise_dist_gpu();


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


#endif

