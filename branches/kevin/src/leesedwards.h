/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_H
#define LEESEDWARDS_H

#include "coords.h"

typedef struct le_s le_t;

__host__ int le_create(coords_t * coords, le_t ** le);
__host__ int le_free(le_t ** le);
__host__ int le_retain(le_t * le);

__host__ int le_nplane_set(le_t * le, int nplanes);
__host__ int le_uy_set(le_t * le, double uy);
__host__ int le_oscillatory_set(le_t * le, int period);
__host__ int le_toffset_set(le_t * le, int nt0);
__host__ int le_info(le_t * le);
__host__ int le_commit(le_t * le);

__host__ int le_comm(le_t * le, MPI_Comm * comm);
__host__ int le_xplane_comm(le_t * le, MPI_Comm * comm);
__host__ int le_jstart_to_mpi_ranks(le_t * le, const int, int send[2], int recv[2]);
__host__ int le_nplane_total(le_t * le, int * npt);
__host__ int le_nplane_local(le_t * le, int * npl);
__host__ int le_uy(le_t * le, double * uy);
__host__ int le_nxbuffer(le_t * le, int * nxb);

/* old static interface */

__host__ MPI_Comm le_communicator(void);
__host__ MPI_Comm le_plane_comm(void);
__host__ int le_jstart_to_ranks(const int, int send[2], int recv[2]);

__host__ int le_get_nxbuffer(void);
__host__ int le_plane_location(const int);
__host__ int le_get_nplane_total(void);
__host__ int le_nsites(void);

__host__ double le_buffer_displacement(const int, const double);
__host__ double le_get_block_uy(int);
__host__ double le_get_steady_uy(const int); 
__host__ double le_plane_uy(const double);
__host__ double le_plane_uy_max(void);
__host__ double le_shear_rate(void);

/* Posible target routines */

__host__ int le_get_nplane_local(void);
__host__ int le_index_real_to_buffer(const int, const int);
__host__ int le_index_buffer_to_real(const int);
__host__ int le_site_index(const int, const int, const int);

#endif
