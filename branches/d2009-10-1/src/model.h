/*****************************************************************************
 *
 *  model.h
 *
 *  $Id: model.h,v 1.14.4.5 2010-02-13 15:40:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _MODEL_H
#define _MODEL_H

#include "d2q9.h"
#include "d3q15.h"
#include "d3q19.h"

#if !defined (_D2Q9_) && !defined (_D3Q15_) && !defined (_D3Q19_)
#error "You must define -D_D2Q9_, -D_D3Q15_ or -D_D3Q19_ in the Makefile" 
#endif

/* Always three dimensions at the moment */
enum {ND = 3};
enum {NHYDRO = 1 + NDIM + NDIM*(NDIM+1)/2};

/* Identify the hydrodynamic modes */
enum {MRHO = 0,
      MRUX = 1, MRUY = 2, MRUZ = 3,
      MSXX = 4, MSXY = 5, MSXZ = 6, MSYY = 7, MSYZ = 8, MSZZ = 9};

extern const double cs2;
extern const double rcs2;
extern struct io_info_t * io_info_distribution_;

void   init_site(void);
void   finish_site(void);
void   halo_site(void);

int    distribution_ndist(void);
void   distribution_ndist_set(const int ndist);
double distribution_f(const int index, const int p, const int n);
void   distribution_f_set(const int index, const int p, const int n,
			  const double fvalue);
double distribution_zeroth_moment(const int index, const int n);
void   distribution_zeroth_moment_set_equilibrium(const int index, const int n,
						  const double rho);
void   distribution_first_moment(const int index, const int n, double g[3]);

void   distribution_get_stress_at_site(int index, double s[ND][ND]);
void   distribution_halo_set_complete(void);
void   distribution_halo_set_reduced(void);


#endif
