/*****************************************************************************
 * 
 * comms_internal_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef COMMS_INTERNAL_GPU_H
#define COMMS_INTERNAL_GPU_H

#include "common_gpu.h"

/* forward declarations of host routines internal to this module */
static void calculate_comms_data_sizes(void);
static void allocate_comms_memory_on_gpu(void);
static void free_comms_memory_on_gpu(void);


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


__global__ static void copy_field_partial_gpu_d(int nPerSite, int nhalo, int N[3],
						double* f_out, double* f_in, char *mask_d, int *packedindex_d, int packedsize, int inpack);

__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);

#endif

