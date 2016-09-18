/*****************************************************************************
 *
 *  lb_model_s.h
 *
 *  LB model data structure implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LB_MODEL_S_H
#define LB_MODEL_S_H

#include "model.h"
#include "halo_swap.h"
#include "io_harness.h"

extern __constant__ int tc_cv[NVEL][3];
extern __constant__ int tc_ndist;

struct lb_data_s {

  int ndist;             /* Number of distributions (default one) */
  int nsite;             /* Number of lattice sites (local) */
  int model;             /* MODEL or MODEL_R */

  pe_t * pe;             /* parallel environment */
  cs_t * cs;             /* coordinate system */
  halo_swap_t * halo;    /* halo swap driver */
  io_info_t * io_info; 

  double * f;            /* Distributions */
  double * fprime;       /* used in propagation only */

  /* MPI data types for halo swaps; these are comupted at runtime
   * to conform to the model selected at compile time */

  MPI_Datatype plane_xy_full;
  MPI_Datatype plane_xz_full;
  MPI_Datatype plane_yz_full;
  MPI_Datatype plane_xy_reduced[2];
  MPI_Datatype plane_xz_reduced[2];
  MPI_Datatype plane_yz_reduced[2];
  MPI_Datatype plane_xy[2];
  MPI_Datatype plane_xz[2];
  MPI_Datatype plane_yz[2];
  MPI_Datatype site_x[2];
  MPI_Datatype site_y[2];
  MPI_Datatype site_z[2];

  lb_t * target;              /* copy of this structure on target */ 
};

/* Data storage: A rank two object */

#define LB_ADDR(nsites, ndist, nvel, index, n, p) \
  addr_rank2(nsites, ndist, nvel, index, n, p)

#endif
