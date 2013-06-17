/*****************************************************************************
 *
 *  model.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef MODEL_H
#define MODEL_H

#include "d2q9.h"
#include "d3q15.h"
#include "d3q19.h"

#if !defined (_D2Q9_) && !defined (_D3Q15_) && !defined (_D3Q19_)
#error "You must define -D_D2Q9_, -D_D3Q15_ or -D_D3Q19_ in the Makefile" 
#endif

#include "io_harness.h"

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

void   distribution_init(void);
void   distribution_finish(void);
void   distribution_halo(void);

int    distribution_ndist(void);
int    distribution_order(void);
void   distribution_ndist_set(const int ndist);
double distribution_f(const int index, const int p, const int n);
void   distribution_f_set(const int index, const int p, const int n,
			  const double fvalue);
double distribution_zeroth_moment(const int index, const int n);
void   distribution_zeroth_moment_set_equilibrium(const int index, const int n,
						  const double rho);
void   distribution_rho_u_set_equilibrium(const int index, const double rho,
					  const double u[3]);
void   distribution_first_moment(const int index, const int n, double g[3]);

void   distribution_get_stress_at_site(int index, double s[3][3]);
void   distribution_halo_set_complete(void);
void   distribution_halo_set_reduced(void);
void   distribution_init_f(void);
void   distribution_index(const int index, const int n, double f[NVEL]);
void distribution_multi_index(const int index, const int n, 
			      double f_vec[NVEL][SIMDVL]);
void distribution_multi_index_part(const int index, const int n, 
				   double f_vec[NVEL][SIMDVL],int nv);
void   distribution_index_set(const int index, const int n,
			      const double f[NVEL]);
void distribution_multi_index_set(const int index, const int n,
				  double f_vec[NVEL][SIMDVL]);
void distribution_multi_index_set_part(const int index, const int n,
				       double f_vec[NVEL][SIMDVL], int nv);

io_info_t * distribution_io_info(void);
void   distribution_io_info_set(io_info_t * io_info);

#endif
