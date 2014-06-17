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


#define MAX_COLLOIDS 500

//#define GPUS_PER_NODE 4
#define GPUS_PER_NODE 1

enum lattchunks {ALL,BULK,EDGES};

/* from coords.h */
enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};
enum upper_triangle {XX, XY, XZ, YY, YZ, ZZ};

//default number of threads per block in each dir
#define DEFAULT_TPB_X 4
#define DEFAULT_TPB_Y 8
#define DEFAULT_TPB_Z 8 

//default number of threads per block
#define DEFAULT_TPB (DEFAULT_TPB_X*DEFAULT_TPB_Y*DEFAULT_TPB_Z) 

/* declarations for required external (host) routines */
extern "C" void coords_nlocal(int n[3]);
extern "C" int coords_nsite_local(int nsite[3]);
extern "C" int coords_nhalo(void);
extern "C" int coords_index(int,int,int);
extern "C" void   coords_index_to_ijk(const int index, int coords[3]);
extern "C" int distribution_ndist(void);
extern "C" void TIMER_start(const int);
extern "C" void TIMER_stop(const int);
extern "C" int le_get_nxbuffer(void);
extern "C" int le_index_real_to_buffer(const int ic, const int di);
extern "C" void TIMER_start(const int);
extern "C" void TIMER_stop(const int);
extern "C" int le_get_nxbuffer(void);
extern "C" int le_index_real_to_buffer(const int ic, const int di);
extern "C" int    phi_nop(void);
extern "C" MPI_Comm cart_comm(void);
extern "C" int    cart_size(const int);
extern "C" int    cart_neighb(const int direction, const int dimension);
extern "C" int    cart_rank(void);
extern "C" int    phi_is_finite_difference(void);
extern "C" int is_propagation_ode();

/* variables for GPU constant memory */
__constant__ int N_cd[3];
__constant__ int Nall_cd[3];
__constant__ int nhalo_cd;
__constant__ int nsites_cd;
__constant__ int nop_cd;


#endif
