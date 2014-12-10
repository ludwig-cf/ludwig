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

int le_create(coords_t * coords, le_t ** le);
int le_free(le_t ** le);
int le_retain(le_t * le);

int le_nplane_set(le_t * le, int nplanes);
int le_uy_set(le_t * le, double uy);
int le_oscillatory_set(le_t * le, int period);
int le_toffset_set(le_t * le, int nt0);
int le_info(le_t * le);
int le_commit(le_t * le);

int le_comm(le_t * le, MPI_Comm * comm);
int le_xplane_comm(le_t * le, MPI_Comm * comm);
int le_jstart_to_mpi_ranks(le_t * le, const int, int send[2], int recv[2]);
int le_nplane_total(le_t * le, int * npt);
int le_nplane_local(le_t * le, int * npl);
int le_uy(le_t * le, double * uy);
int le_nxbuffer(le_t * le, int * nxb);

/* old static interface */

MPI_Comm le_communicator(void);
MPI_Comm le_plane_comm(void);
int le_jstart_to_ranks(const int, int send[2], int recv[2]);

int le_get_nxbuffer(void);
int le_plane_location(const int);
int le_get_nplane_total(void);
int le_nsites(void);

double le_buffer_displacement(const int, const double);
double le_get_block_uy(int);
double le_get_steady_uy(const int); 
double le_plane_uy(const double);
double le_plane_uy_max(void);
double le_shear_rate(void);

/* Posible target routines */

int le_get_nplane_local(void);
int le_index_real_to_buffer(const int, const int);
int le_index_buffer_to_real(const int);
int le_site_index(const int, const int, const int);

#endif
