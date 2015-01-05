/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_H
#define LEESEDWARDS_H

#include "coords.h"

typedef struct le_s le_t;

__host__ int le_create(coords_t * coords, le_t ** le);
__host__ int le_free(le_t * le);
__host__ int le_retain(le_t * le);

__host__ int le_nplane_set(le_t * le, int nplanes);
__host__ int le_plane_uy_set(le_t * le, double uy);
__host__ int le_oscillatory_set(le_t * le, int period);
__host__ int le_toffset_set(le_t * le, int nt0);
__host__ int le_info(le_t * le);
__host__ int le_commit(le_t * le);
__host__ int le_comm(le_t * le, MPI_Comm * comm);
__host__ int le_plane_comm(le_t * le, MPI_Comm * comm);
__host__ int le_jstart_to_mpi_ranks(le_t * le, int, int send[2], int recv[2]);

/* coords 'inherited' interface host / device */

__host__ int le_nhalo(le_t * le, int * nhalo);
__host__ int le_nsites(le_t * le, int * nsites);
__host__ int le_nlocal(le_t * le, int nlocal[3]);
__host__ int le_site_index(le_t * le, int ic, int jc, int kc);
__host__ int le_strides(le_t * le, int * xs, int * ys, int * zs);
__host__ int le_ltot(le_t * le, double ltot[3]);
__host__ int le_cartsz(le_t * le, int cartsz[3]);
__host__ int le_ntotal(le_t * le, int ntotal[3]);
__host__ int le_nlocal_offset(le_t * le, int offset[3]);
__host__ int le_cart_coords(le_t * le, int cartcoords[3]);

/* Additional host/target routines */

__host__ int le_index_real_to_buffer(le_t * le, int ic, int idisplace);
__host__ int le_index_buffer_to_real(le_t * le, int ibuf);
__host__ int le_nplane_total(le_t * le, int * npt);
__host__ int le_nplane_local(le_t * le, int * npl);
__host__ int le_plane_uy(le_t * le, double * uy);
__host__ int le_plane_uy_now(le_t * le, double t, double * uy);
__host__ int le_nxbuffer(le_t * le, int * nxb);
__host__ int le_shear_rate(le_t * le, double * gammadot);
__host__ int le_steady_uy(le_t * le, int ic, double * uy); 
__host__ int le_plane_location(le_t * le, int plane);
__host__ int le_buffer_displacement(le_t * le, int ib, double t, double * dy);
__host__ int le_get_block_uy(le_t * le, int , double * uy);

#endif
