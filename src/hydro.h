/*****************************************************************************
 *
 *  hydro.h
 *
 *  Various hydrodynamic-related quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_HYDRO_H
#define LUDWIG_HYDRO_H

#include "pe.h"
#include "coords.h"
#include "halo_swap.h"
#include "leesedwards.h"
#include "io_harness.h"
#include "hydro_options.h"

/* Halo */

#include "cs_limits.h"

typedef struct hydro_halo_s hydro_halo_t;

struct hydro_halo_s {

  MPI_Comm comm;                /* coords: Cartesian communicator */
  int nbrrank[3][3][3];         /* coords: Cartesian neighbours */

  int nvel;                     /* Number of directions involved (2d or 3d) */
  int8_t cv[27][3];             /* Send/recv directions */
  cs_limits_t slim[27];         /* halo: send regions (rectangular) */
  cs_limits_t rlim[27];         /* halo: recv regions (rectangular) */
  double * send[27];            /* halo: send data buffers */
  double * recv[27];            /* halo: recv data buffers */
  MPI_Request request[2*27];    /* halo: array of send/recv requests */
};

typedef struct hydro_s hydro_t;

/* Data storage: Always a 3-vector NHDIM */

#define NHDIM 3

struct hydro_s {
  int nsite;               /* Allocated sites (local) */
  int nhcomm;              /* Width of halo region for u field */
  double * rho;            /* Density field */
  double * u;              /* Velocity field (on host) */
  double * f;              /* Body force field (on host) */
  double * eta;            /* Local shear stress */

  pe_t * pe;               /* Parallel environment */
  cs_t * cs;               /* Coordinate system */
  lees_edw_t * le;         /* Lees Edwards */
  io_info_t * info;        /* I/O handler. */
  halo_swap_t * halo;      /* Halo driver object */

  hydro_halo_t h;          /* Host OpenMP halo swap */
  hydro_options_t opts;    /* Copy of the options requested */

  hydro_t * target;        /* structure on target */ 
};

__host__ int hydro_create(pe_t * pe, cs_t * cs, lees_edw_t * le, 
			  const hydro_options_t * opts, hydro_t ** pobj);
__host__ int hydro_free(hydro_t * obj);
__host__ int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int hydro_memcpy(hydro_t * ibj, tdpMemcpyKind flag);
__host__ int hydro_io_info(hydro_t * obj, io_info_t ** info);
__host__ int hydro_u_halo(hydro_t * obj);
__host__ int hydro_halo_swap(hydro_t * obj, hydro_halo_enum_t flag);
__host__ int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
				     double w[3][3]);
__host__ int hydro_lees_edwards(hydro_t * obj);
__host__ int hydro_correct_momentum(hydro_t * obj);
__host__ int hydro_f_zero(hydro_t * obj, const double fzero[3]);
__host__ int hydro_u_zero(hydro_t * obj, const double uzero[3]);
__host__ int hydro_rho0(hydro_t * hydro, double rho0);

__host__ __device__ int hydro_f_local_set(hydro_t * obj, int index,
					  const double force[3]);
__host__ __device__ int hydro_f_local(hydro_t * obj, int index, double force[3]);
__host__ __device__ int hydro_f_local_add(hydro_t * obj, int index,
					  const double force[3]);
__host__ __device__ int hydro_rho_set(hydro_t * hydro, int index, double rho);
__host__ __device__ int hydro_rho(hydro_t * hydro, int index, double * rho);
__host__ __device__ int hydro_u_set(hydro_t * obj, int index, const double u[3]);
__host__ __device__ int hydro_u(hydro_t * obj, int index, double u[3]);

/* Host halo interface */

int hydro_halo_create(const hydro_t * hydro, hydro_halo_t * h);
int hydro_halo_post(hydro_t * hydro);
int hydro_halo_wait(hydro_t * hydro);
int hydro_halo_free(hydro_halo_t * h);

#endif
