/*****************************************************************************
 *
 *  model.h
 *
 *  This includes the appropriate lattice Boltzmann
 *  model d2q9, d3q15, or d3q19 (see Makefile).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef MODEL_H
#define MODEL_H

/* Here is the choice of model (and it's here only). */

#ifdef _D2Q9_
#include "d2q9.h"
#endif

#ifdef _D3Q15_
#include "d3q15.h"
#endif

#ifdef _D3Q19_
#include "d3q19.h"
#endif

#include "io_harness.h"
#include "targetDP.h"

/* Vector length for SIMD auto-vectorisation over lattice sites. */
/* If not set in the Makefile, it defaults to 1, as larger values
 * can result in adverse performance (e.g., if choice doesn't
 * match hardware, or in 2d) */

#if !defined (SIMDVL)
#define SIMDVL 1
#endif


/* Number of hydrodynamic modes */
enum {NHYDRO = 1 + NDIM + NDIM*(NDIM+1)/2};

/* Memory arrangement */
enum {MODEL, MODEL_R};

extern const double cs2;
extern const double rcs2;

typedef enum lb_dist_enum_type{LB_RHO = 0, LB_PHI = 1} lb_dist_enum_t;

typedef struct lb_data_s lb_t;
typedef enum lb_halo_enum_type {LB_HALO_FULL, LB_HALO_REDUCED} lb_halo_enum_t;

__targetHost__ int lb_create_ndist(int ndist, lb_t ** lb);
__targetHost__ int lb_create(lb_t ** lb);
__targetHost__ void lb_free(lb_t * lb);
__targetHost__ int lb_nvel(lb_t * lb, int * nvel);
__targetHost__ int lb_ndim(lb_t * lb, int * ndim);
__targetHost__ int lb_nblock(lb_t * lb, int dim, int * nblock);
__targetHost__ int lb_init(lb_t * lb);
__targetHost__ int lb_init_rest_f(lb_t * lb, double rho0);
__targetHost__ int lb_halo(lb_t * lb);
__targetHost__ int lb_halo_via_copy(lb_t * lb);
__targetHost__ int lb_halo_via_struct(lb_t * lb);
__targetHost__ int lb_ndist(lb_t * lb, int * ndist);
__targetHost__ int lb_order(lb_t * lb);
__targetHost__ int lb_ndist_set(lb_t * lb, int ndist);
__targetHost__ int lb_f(lb_t * lb, int index, int p, int n, double * f);
__targetHost__ int lb_f_set(lb_t * lb, int index, int p, int n, double fval);
__targetHost__ int lb_0th_moment(lb_t * lb, int index, lb_dist_enum_t nd, double * rho);
__targetHost__ int lb_1st_moment(lb_t * lb, int index, lb_dist_enum_t nd, double g[3]);
__targetHost__ int lb_2nd_moment(lb_t * lb, int index, lb_dist_enum_t nd, double s[3][3]);
__targetHost__ int lb_0th_moment_equilib_set(lb_t * lb, int index, int n, double rho);
__targetHost__ int lb_1st_moment_equilib_set(lb_t * lb, int index, double rho, double u[3]);
__targetHost__ int lb_halo_set(lb_t * lb, lb_halo_enum_t halo);
__targetHost__ int lb_halo_reduced(lb_t * lb);
__targetHost__ int lb_f_index(lb_t * lb, int index, int n, double f[NVEL]);
__targetHost__ int lb_f_multi_index(lb_t * lb, int index, int n, double f[NVEL][SIMDVL]);
__targetHost__ int lb_f_index_set(lb_t * lb, int index, int n, double f[NVEL]);
__targetHost__ int lb_f_multi_index_set(lb_t * lb, int index, int n, double f[NVEL][SIMDVL]);
__targetHost__ int lb_f_multi_index_part(lb_t * lb, int index, int n, double f[NVEL][SIMDVL],
			  int nv);
__targetHost__ int lb_f_multi_index_set_part(lb_t * lb, int index, int n,
			      double f[NVEL][SIMDVL], int nv);
__targetHost__ int lb_io_info(lb_t * lb, io_info_t ** io_info);
__targetHost__ int lb_io_info_set(lb_t * lb, io_info_t * io_info);

#endif
