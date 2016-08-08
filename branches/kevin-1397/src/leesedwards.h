/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_H
#define LEESEDWARDS_H

#include "pe.h"
#include "runtime.h"
#include "coords.h"

typedef struct lees_edw_s lees_edw_t;

__host__ int le_create(pe_t * pe, coords_t * cs, lees_edw_t ** ple);
__host__ int le_free(lees_edw_t * le);

__targetHost__ int le_init(pe_t * pe, rt_t * rt);
__targetHost__ void le_finish(void);

__targetHost__ int le_info(void);
__targetHost__ int le_get_nxbuffer(void);
__targetHost__ int le_index_buffer_to_real(const int);
__targetHost__ int le_plane_location(const int);
__targetHost__ int le_get_nplane_total(void);
__targetHost__ int le_get_nplane_local(void);
__targetHost__ int le_nall(int nall[3]);

__host__ __device__ int le_index_real_to_buffer(const int, const int);
__host__ __device__ int le_nsites(void);
__host__ __device__ int le_site_index(const int, const int, const int);

__targetHost__ double    le_buffer_displacement(const int, const double);
__host__ int le_buffer_dy(int ib, double * dy);
__host__ int le_buffer_du(int ib, double ule[3]);
__targetHost__ double    le_get_block_uy(int);
__targetHost__ double    le_get_steady_uy(const int); 
__targetHost__ double    le_plane_uy(const double);
__host__ int le_plane_dy(double * dy);

__targetHost__ double    le_plane_uy_max(void);
__targetHost__ double    le_shear_rate(void);
__targetHost__ MPI_Comm  le_communicator(void);
__targetHost__ MPI_Comm  le_plane_comm(void);
__targetHost__ void      le_jstart_to_ranks(const int, int send[2], int recv[2]);
__targetHost__ void      le_set_oscillatory(const double);
__targetHost__ void      le_set_nplane_total(const int nplane);
__targetHost__ void      le_set_plane_uymax(const double uy);

#endif
