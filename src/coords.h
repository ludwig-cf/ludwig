/*****************************************************************************
 *
 *  coords.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COORDS_H
#define LUDWIG_COORDS_H

#include "pe.h"
#include "cartesian.h"
#include "lees_edwards_options.h"
#include "util_json.h"

/* Note that there is a copy of the Lees Edwards options here to
 * allow the relevant metadata to be constructed from coords_t alone.
 * All the Lees Edwards operations are in leesedwards.c */

typedef struct coords_s cs_t;
typedef struct coords_param_s cs_param_t;

struct coords_param_s {
  int nhalo;                       /* Width of halo region */
  int nsites;                      /* Total sites (incl. halo) */
  int ntotal[3];                   /* System (physical) size */
  int nlocal[3];                   /* Local system size */
  int noffset[3];                  /* Local system offset */
  int str[3];                      /* Memory strides */
  int periodic[3];                 /* Periodic boundary (non-periodic = 0) */

  int mpi_cartsz[3];               /* Cartesian size */
  int mpi_cartcoords[3];           /* Cartesian coordinates lookup */

  double lmin[3];                  /* System L_min */
};

struct coords_s {
  pe_t * pe;                       /* Retain a reference to pe */
  int nref;                        /* Reference count */

  cs_param_t * param;              /* Constants */

  /* Host data */
  int mpi_cartrank;                /* MPI Cartesian rank */
  int reorder;                     /* MPI reorder flag */
  int mpi_cart_neighbours[2][3];   /* Ranks of Cartesian neighbours lookup */
  int * listnlocal[3];             /* Rectilinear decomposition */
  int * listnoffset[3];            /* Rectilinear offsets */
  lees_edw_options_t leopts;       /* Copy of LE opts (no. of planes etc.) */

  MPI_Comm commcart;               /* Cartesian communicator */
  MPI_Comm commperiodic;           /* Cartesian periodic communicator */

  cs_t * target;                   /* Host pointer to target memory */
};

#define NSYMM 6      /* Elements for general symmetric tensor */

enum cartesian_neighbours {FORWARD, BACKWARD};
enum cs_mpi_cart_neighbours {CS_FORW=0,CS_BACK=1};
enum upper_triangle {XX, XY, XZ, YY, YZ, ZZ};

/* Host interface */

__host__ int cs_create(pe_t * pe, cs_t ** pcs);
__host__ int cs_free(cs_t * cs);
__host__ int cs_retain(cs_t * cs);
__host__ int cs_init(cs_t * cs);
__host__ int cs_commit(cs_t * cs);
__host__ int cs_target(cs_t * cs, cs_t ** target);

__host__ int cs_decomposition_set(cs_t * cs, const int irequest[3]);
__host__ int cs_periodicity_set(cs_t * cs, const int iper[3]);
__host__ int cs_ntotal_set(cs_t * cs, const int ntotal[3]);
__host__ int cs_nhalo_set(cs_t * cs, int nhalo);
__host__ int cs_reorder_set(cs_t * cs, int reorder);
__host__ int cs_info(cs_t * cs);
__host__ int cs_cart_comm(cs_t * cs, MPI_Comm * comm);
__host__ int cs_periodic_comm(cs_t * cs, MPI_Comm * comm);
__host__ int cs_cart_neighb(cs_t * cs, int forwback, int dim);
__host__ int cs_cart_rank(cs_t * cs);
__host__ int cs_pe_rank(cs_t * cs);

/* Host / device interface */

__host__ __device__ int cs_cartsz(cs_t * cs, int cartsz[3]);
__host__ __device__ int cs_cart_coords(cs_t * cs, int coords[3]);
__host__ __device__ int cs_lmin(cs_t * cs, double lmin[3]);
__host__ __device__ int cs_ltot(cs_t * cs, double ltot[3]);
__host__ __device__ int cs_periodic(cs_t * cs, int period[3]);
__host__ __device__ int cs_nlocal(cs_t * cs, int n[3]);
__host__ __device__ int cs_nlocal_offset(cs_t * cs, int n[3]);
__host__ __device__ int cs_nhalo(cs_t *cs, int * nhalo);
__host__ __device__ int cs_index(cs_t * cs, int ic, int jc, int kc);
__host__ __device__ int cs_ntotal(cs_t * cs, int ntotal[3]);
__host__ __device__ int cs_nsites(cs_t * cs, int * nsites);
__host__ __device__ int cs_minimum_distance(cs_t * cs, const double r1[3],
			    const double r2[3], double r12[3]);
__host__ __device__ int cs_index_to_ijk(cs_t * cs, int index, int coords[3]);
__host__ __device__ int cs_strides(cs_t * cs, int *xs, int *ys, int *zs);
__host__ __device__ int cs_nall(cs_t * cs, int nall[3]);

/* A "class" function */

__host__ int cs_cart_shift(MPI_Comm comm, int dim, int direction, int * rank);

int cs_options_to_json(const cs_param_t * opts, cJSON ** json);
int cs_options_from_json(const cJSON * json, cs_param_t * opts);
int cs_to_json(const cs_t * cs, cJSON ** json);
int cs_from_json(pe_t * pe, const cJSON * json, cs_t ** cs);

#endif
