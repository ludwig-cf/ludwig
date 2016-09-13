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

#ifndef LUDWIG_LEES_EDWARDS_H
#define LUDWIG_LEES_EDWARDS_H

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "physics.h"

typedef struct lees_edw_s lees_edw_t;
typedef struct lees_edw_info_s lees_edw_info_t;

struct lees_edw_info_s {
  int nplanes;
  int type;
  int period;
  int nt0;
  double uy;
};

typedef enum lees_edw_enum {LE_SHEAR_TYPE_STEADY,
			    LE_SHEAR_TYPE_OSCILLATORY} lees_edw_enum_t;


__host__ int lees_edw_create(pe_t * pe, cs_t * coords, lees_edw_info_t * info,
			     lees_edw_t ** le);
__host__ int lees_edw_free(lees_edw_t * le);
__host__ int lees_edw_retain(lees_edw_t * le);
__host__ int lees_edw_info(lees_edw_t * le);
__host__ int lees_edw_comm(lees_edw_t * le, MPI_Comm * comm);
__host__ int lees_edw_plane_comm(lees_edw_t * le, MPI_Comm * comm);
__host__ int lees_edw_jstart_to_mpi_ranks(lees_edw_t * le, int, int send[2], int recv[2]);
__host__ int lees_edw_buffer_dy(lees_edw_t * le, int ib, double t0, double * dy);
__host__ int lees_edw_buffer_du(lees_edw_t * le, int ib, double ule[3]);


/* coords 'inherited' interface host / device */

__host__ __device__ int lees_edw_nhalo(lees_edw_t * le, int * nhalo);
__host__ __device__ int lees_edw_nsites(lees_edw_t * le, int * nsites);
__host__ __device__ int lees_edw_nlocal(lees_edw_t * le, int nlocal[3]);
__host__ __device__ int lees_edw_index(lees_edw_t * le, int ic, int jc, int kc);
__host__ __device__ int lees_edw_strides(lees_edw_t * le, int * xs, int * ys, int * zs);
__host__ __device__ int lees_edw_ltot(lees_edw_t * le, double ltot[3]);
__host__ __device__ int lees_edw_cartsz(lees_edw_t * le, int cartsz[3]);
__host__ __device__ int lees_edw_ntotal(lees_edw_t * le, int ntotal[3]);
__host__ __device__ int lees_edw_nlocal_offset(lees_edw_t * le, int offset[3]);
__host__ __device__ int lees_edw_cart_coords(lees_edw_t * le, int cartcoords[3]);

/* Additional host / device routines */

__host__ __device__ int lees_edw_index_real_to_buffer(lees_edw_t * le, int ic, int idisplace);
__host__ __device__ int lees_edw_index_buffer_to_real(lees_edw_t * le, int ibuf);
__host__ __device__ int lees_edw_nplane_total(lees_edw_t * le);
__host__ __device__ int lees_edw_nplane_local(lees_edw_t * le);
__host__ __device__ int lees_edw_plane_uy(lees_edw_t * le, double * uy);
__host__ __device__ int lees_edw_plane_uy_now(lees_edw_t * le, double t, double * uy);
__host__ __device__ int lees_edw_plane_dy(lees_edw_t * le, double * dy);
__host__ __device__ int lees_edw_nxbuffer(lees_edw_t * le, int * nxb);
__host__ __device__ int lees_edw_shear_rate(lees_edw_t * le, double * gammadot);
__host__ __device__ int lees_edw_steady_uy(lees_edw_t * le, int ic, double * uy); 
__host__ __device__ int lees_edw_plane_location(lees_edw_t * le, int plane);
__host__ __device__ int lees_edw_buffer_displacement(lees_edw_t * le, int ib, double t, double * dy);
__host__ __device__ int lees_edw_block_uy(lees_edw_t * le, int , double * uy);

#endif
