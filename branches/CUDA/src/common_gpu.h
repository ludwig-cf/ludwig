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

/* from coords.h */
enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};
enum upper_triangle {XX, XY, XZ, YY, YZ, ZZ};

#define DEFAULT_TPB 256 //default number of threads per bock

/* declarations for required external (host) routines */
extern "C" void coords_nlocal(int n[3]);
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

#endif
