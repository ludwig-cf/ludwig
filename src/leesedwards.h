/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2022 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_LEES_EDWARDS_H
#define LUDWIG_LEES_EDWARDS_H

#include "pe.h"
#include "memory.h"
#include "runtime.h"
#include "coords.h"
#include "physics.h"
#include "lees_edwards_options.h"

typedef struct lees_edw_s lees_edw_t;
typedef struct lees_edw_param_s lees_edw_param_t;

struct lees_edw_s {
    pe_t *pe;        /* Parallel environment */
    cs_t *cs;        /* Coordinate system */
    physics_t *phys; /* Constants, time step */

    lees_edw_param_t *param; /* Parameters */

    int nref;            /* Reference count */
    int *icbuff_to_real; /* look up table */
    int *icreal_to_buff; /* look up table */
    int *buffer_duy;     /* look up table +/- uy as function of ib */

    MPI_Comm le_comm;       /* 1-d communicator */
    MPI_Comm le_plane_comm; /* 2-d communicator */

    lees_edw_t *target; /* Device memory */
};

struct lees_edw_param_s {
    /* Local parameters */
    int nplanelocal; /* Number of planes local domain */
    int nxbuffer;    /* Size of buffer region in x */
    int index_real_nbuffer;
    /* For cs */
    int nhalo;
    int str[3];
    int nlocal[3];
    /* Global parameters */
    int nplanetotal; /* Total number of planes */
    int type;        /* Shear type */
    int period;      /* for oscillatory */
    int nt0;         /* time0 (input as integer) */
    int nsites;      /* Number of sites incl buffer planes */
    double uy;       /* u[Y] for all planes */
    double dx_min;   /* Position first plane */
    double dx_sep;   /* Plane separation */
    double omega;    /* u_y = u_le cos (omega t) for oscillatory */
    double time0;    /* time offset */
};
__host__ int lees_edw_create(pe_t * pe, cs_t * coords,
			     const lees_edw_options_t * opts,
			     lees_edw_t ** le);
__host__ int lees_edw_free(lees_edw_t * le);
__host__ int lees_edw_retain(lees_edw_t * le);
__host__ int lees_edw_commit(lees_edw_t * le);
__host__ int lees_edw_target(lees_edw_t * le, lees_edw_t ** target);

__host__ int lees_edw_info(lees_edw_t * le);
__host__ int lees_edw_comm(lees_edw_t * le, MPI_Comm * comm);
__host__ int lees_edw_plane_comm(lees_edw_t * le, MPI_Comm * comm);
__host__ int lees_edw_jstart_to_mpi_ranks(lees_edw_t * le, int, int send[3], int recv[3]);
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

__host__ __device__ int lees_edw_ibuff_to_real(lees_edw_t * le, int ib);
__host__ __device__ int lees_edw_ic_to_buff(lees_edw_t * le, int ic, int di);

__host__ __device__ void lees_edw_index_v(lees_edw_t * le, int ic[NSIMDVL],
					  int jc[NSIMDVL], int kc[NSIMDVL],
					  int index[NSIMDVL]);
#endif
