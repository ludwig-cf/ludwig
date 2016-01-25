/*****************************************************************************
 *
 *  lb_model_s.h
 *
 *  LB model data structure implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LB_MODEL_S_H
#define LB_MODEL_S_H

#include "model.h"
#include "io_harness.h"

/* Data storage */
/* A preprocessor macro is provided to switch between two options
 * for the arrangement of the distributions in memory:
 *
 *   MODEL_R is 'structure of arrays'
 *   MODEL   is 'array of structures'
 *
 * The following macros  allow the distribution to ba addressed in
 * terms of:
 *
 *  lattice spatial index = coords_index(ic, jc, kc) 0 ... nsite
 *  distribution n (LB_RHO and optionally LB_PHI)    0 ... ndist
 *  discrete velocity p                              0 ... NVEL
 */

/* Distribution 'array of structures' for 'MODEL' (independent of nsite) */
#define LB_ADDR_MODEL(nsite, ndist, nvel, index, n, p) \
  ((ndist)*nvel*(index) + (n)*nvel + (p))

/* Distribution 'structure of arrays' for 'MODEL_R' (independent of NVEL) */
#define LB_ADDR_MODEL_R(nsite, ndist, nvel, index, n, p) \
  ((p)*ndist*nsite + (n)*nsite + (index))

/* Distribution 'array of structure of short arrays' */
#define SAN VVL
#define LB_ADDR_MODEL_AoSoA(nsite, ndist, nvel, index, n, p) \
  (((index)/SAN)*(ndist)*(nvel)*SAN + (n)*(nvel)*SAN + (p)*SAN + ((index)-((index)/SAN)*SAN))


#ifdef LB_DATA_SOA
#define LB_DATA_MODEL MODEL_R

#ifdef AOSOA
#define LB_ADDR LB_ADDR_MODEL_AoSoA
#else
#define LB_ADDR LB_ADDR_MODEL_R
#endif

#else
#define LB_DATA_MODEL MODEL
#define LB_ADDR LB_ADDR_MODEL
#endif

struct lb_data_s {

  int ndist;             /* Number of distributions (default one) */
  int nsite;             /* Number of lattice sites (local) */
  int model;             /* MODEL or MODEL_R */
  io_info_t * io_info; 

  double * f;            /* Distributions (on host)*/
  double * fprime;            /* data staging space */

  double * t_f;            /* Distributions (on target)*/

  double * t_fprime;            /* data staging space (on target)*/


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

  lb_t * tcopy;              /* copy of this structure on target */ 

};

extern __targetConst__ int tc_cv[NVEL][3];
extern __targetConst__ int tc_ndist;

#endif
