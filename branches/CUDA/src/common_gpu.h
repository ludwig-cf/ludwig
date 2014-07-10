/*****************************************************************************
 * 
 * common_gpu.h
 * 
 * Common headers for GPU adaptation of Ludwig
 * Alan Gray
 *
 *****************************************************************************/

#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include "pe.h"
#include "coords.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_COLLOIDS 500

//#define GPUS_PER_NODE 4
#define GPUS_PER_NODE 1

enum lattchunks {ALL,BULK,EDGES};

//default number of threads per block in each dir
#define DEFAULT_TPB_X 4
#define DEFAULT_TPB_Y 8
#define DEFAULT_TPB_Z 8

//default number of threads per block
#define DEFAULT_TPB (DEFAULT_TPB_X*DEFAULT_TPB_Y*DEFAULT_TPB_Z) 

/* declarations for required external (host) routines */

int distribution_ndist(void);
void TIMER_start(const int);
void TIMER_stop(const int);
int le_get_nxbuffer(void);
int le_index_real_to_buffer(const int ic, const int di);
int    phi_nop(void);
int    phi_is_finite_difference(void);
int is_propagation_ode();

/* variables for GPU constant memory */
__constant__ int N_cd[3];
__constant__ int Nall_cd[3];
__constant__ int nhalo_cd;
__constant__ int nsites_cd;
__constant__ int nop_cd;


#ifdef __cplusplus
}
#endif

#endif
