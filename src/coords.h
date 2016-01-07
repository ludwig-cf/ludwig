/*****************************************************************************
 *
 *  coords.h
 *
 *  $Id: coords.h,v 1.4 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COORDS_H
#define COORDS_H

#include <mpi.h>

#include "targetDP.h"

#define NSYMM 6      /* Elements for general symmetric tensor */

enum cartesian_directions {X, Y, Z};
enum cartesian_neighbours {FORWARD, BACKWARD};
enum upper_triangle {XX, XY, XZ, YY, YZ, ZZ};

__targetHost__ void   coords_init(void);
__targetHost__ void   coords_finish(void);
__targetHost__ void   coords_info(void);
__targetHost__ int    N_total(const int);
__targetHost__ int    is_periodic(const int);
__targetHost__ double L(const int);
__targetHost__ double Lmin(const int);
__targetHost__ int    cart_rank(void);
__targetHost__ int    cart_size(const int);
__targetHost__ int    cart_coords(const int);
__targetHost__ int    cart_neighb(const int direction, const int dimension);

__targetHost__ MPI_Comm cart_comm(void);

__targetHost__ void   coords_nlocal(int n[3]);
__targetHost__ void   coords_nlocal_offset(int n[3]);
__targetHost__ void   coords_nhalo_set(const int nhalo);
__targetHost__ int    coords_nhalo(void);
__targetHost__ int    coords_ntotal(int ntotal[3]);
__targetHost__ void   coords_ntotal_set(const int n[3]);
__targetHost__ void   coords_decomposition_set(const int p[3]);
__targetHost__ void   coords_reorder_set(const int);
__targetHost__ void   coords_periodicity_set(const int p[3]);
__targetHost__ int    coords_nsites(void);
__targetHost__ int    coords_index(const int ic, const int jc, const int kc);
__targetHost__ void   coords_minimum_distance(const double r1[3], const double r2[3],
			       double r12[3]);
__targetHost__ void   coords_index_to_ijk(const int index, int coords[3]);
__targetHost__ int    coords_strides(int * xs, int * ys, int * zs);

void coords_active_region_radius_set(const double r);
__targetHost__ double coords_active_region(const int index);
int    coords_periodic_comm(MPI_Comm * comm);
int    coords_cart_shift(MPI_Comm comm, int dim, int direction, int * rank);

extern __targetConst__ int tc_nSites;
extern __targetConst__ int tc_noffset[3]; 
extern __targetConst__ int tc_Nall[3];
extern __targetConst__ int tc_nhalo;
extern __targetConst__ int tc_nextra;


/* memory addressing macro for 1d vectors on each site */
/* (e.g. force, velocity, ...)*/
/* A preprocessor macro is provided to switch between two options
 * for the arrangement of grid-based field objects in memory:
 *
 * The following macros allow the objects to be addressed in
 * terms of:
 *
 *  lattice spatial index = coords_index(ic, jc, kc) 0 ... nsite
 *  field index ifield                               0 ... nfield
 */

/* array of structures */
#define ADDR_VECSITE(nsite, nfield, index, ifield)	\
  ((nfield)*(index) + (ifield))


/* structure of arrays */
#define ADDR_VECSITE_R(nsite, nfield, index, ifield)	\
  ((nsite)*(ifield) + (index))

#ifdef LB_DATA_SOA
#define VECADR ADDR_VECSITE_R
#else
#define VECADR ADDR_VECSITE
#endif


/* memory addressing macro for 3x3 fields. */

/* array of structures */
#define ADDR_3X3(nsite, index, ia, ib)	\
  (3*3*(index) + 3*(ia)+(ib))


/* structure of arrays */
#define ADDR_3X3_R(nsite, index, ia, ib)	\
  (3*(nsite)*(ia)+(nsite)*(ib)+(index))

#ifdef LB_DATA_SOA
#define ADR3X3 ADDR_3X3_R
#else
#define ADR3X3 ADDR_3X3
#endif

/* legacy */
#define PTHADR ADR3X3

/* memory addressing macro for 3x3x3 fields. */

/* array of structures */
#define ADDR_3X3X3(nsite, index, ia, ib, ic)	\
  (3*3*3*(index)+3*3*(ia)+3*(ib)+(ic))


/* structure of arrays */
#define ADDR_3X3X3_R(nsite, index, ia, ib, ic)	\
  (3*3*(nsite)*(ia)+3*(nsite)*(ib)+(nsite)*(ic)+(index))

#ifdef LB_DATA_SOA
#define ADR3X3X3 ADDR_3X3X3_R
#else
#define ADR3X3X3 ADDR_3X3X3
#endif




#endif
