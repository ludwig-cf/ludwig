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
#include "model.h"

/* expose routines in this module to outside routines */
extern "C" void put_f_on_gpu();
extern "C" void get_f_from_gpu();
extern "C" void put_f_partial_on_gpu(int *mask, int include_neighbours);
extern "C" void get_f_partial_from_gpu(int *mask, int include_neighbours);
extern "C" void put_velocity_partial_on_gpu(int include_neighbours);
extern "C" void get_velocity_partial_from_gpu(int include_neighbours);
extern "C" void update_colloid_force_from_gpu();
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void distribution_halo_gpu(void);
extern "C" void copy_f_to_ftmp_on_gpu(void);
extern "C" void get_f_edges_from_gpu(void);
extern "C" void put_f_halos_on_gpu(void);
extern "C" void bounce_back_gpu(int *findexall, int *linktype,
				double *dfall, double *dgall1, double *dgall2,
				double *dmall, int nlink, int pass);
extern "C" void bbl_init_temp_link_arrays_gpu(int nlink);
extern "C" void bbl_finalise_temp_link_arrays_gpu();
extern "C" void bbl_enlarge_temp_link_arrays_gpu(int nlink);
extern "C" void halo_gpu(int nfields1, int nfields2, int packfield1, double * data_d);

/* forward declarations of host routines internal to this module */
static void calculate_dist_data_sizes(void);
static void allocate_dist_memory_on_gpu(void);
static void free_dist_memory_on_gpu(void);
void init_dist_gpu();
void finalise_dist_gpu();


/* forward declarations of accelerator routines internal to this module */
__global__ static void pack_edge_gpu_d(int nfields1, int nfields2,
				       int nhalo, int nreduced,
					 int N[3],
					 double* fedgeLOW_d,
				       double* fedgeHIGH_d, 
				       double* f_d, int dirn);

__global__ static void unpack_halo_gpu_d(int nfields1, int nfields2,
					 int nhalo, int nreduced,
					   int N[3],
					   double* f_d, double* fhaloLOW_d,
					 double* fhaloHIGH_d, int dirn);



/* external variables holding device memory addresses */
extern double * phi_site_d;
extern double * colloid_force_d;


/* constant memory symbols internal to this module */
__constant__ int cv_cd[NVEL][3];
#endif

