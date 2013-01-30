/*****************************************************************************
 * 
 * comms_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef DISTUTILITIES_GPU_H
#define DISTUTILITIES_GPU_H

#include "common_gpu.h"
#include "model.h"

/* expose routines in this module to outside routines */
extern "C" void halo_gpu(int nfields1, int nfields2, int packfield1, double * data_d);
extern "C" void put_field_partial_on_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));

extern "C" void get_field_partial_from_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));

/* forward declarations of host routines internal to this module */
static void calculate_comms_data_sizes(void);
static void allocate_comms_memory_on_gpu(void);
static void free_comms_memory_on_gpu(void);
void init_comms_gpu();
void finalise_comms_gpu();


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
						double* f_out, double* f_in, int *mask_d, int *packedindex_d, int packedsize, int inpack);


__global__ static void copy_field_partial_gpu_d_TEST(int nPerSite, int nhalo, int N[3],
						double* f_out, double* f_in, int *mask_d, int *packedindex_d, int packedsize, int inpack);





/* constant memory symbols internal to this module */
__constant__ int cv_cd[NVEL][3];
#endif

