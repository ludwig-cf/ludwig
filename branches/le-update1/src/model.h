/*****************************************************************************
 *
 *  model.h
 *
 *  $Id: model.h,v 1.11 2008-08-24 16:12:44 kevin Exp $
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

#include "d3q15.h"
#include "d3q19.h"

#if !defined (_D3Q15_) && !defined (_D3Q19_)
#error "You must define either -D_D3Q15_ or -D_D3Q19_ in the Makeifle" 
#endif

/* Always three dimensions at the moment */
enum {ND = 3};
enum {NHYDRO = 10};

/* Identify the hydrodynamic modes */
enum {MRHO = 0,
      MRUX = 1, MRUY = 2, MRUZ = 3,
      MSXX = 4, MSXY = 5, MSXZ = 6, MSYY = 7, MSYZ = 8, MSZZ = 9};

typedef struct {
  double f[NVEL], g[NVEL];
} Site;

extern const double cs2;
extern const double rcs2;
extern const double d_[3][3];
extern struct io_info_t * io_info_distribution_;

void   init_site(void);
void   finish_site(void);
void   halo_site(void);

double get_f_at_site(const int, const int);
double get_g_at_site(const int, const int);
double get_rho_at_site(const int);
double get_phi_at_site(const int);
void   set_rho(const double, const int);
void   set_phi(const double, const int);
void   set_f_at_site(const int, const int, const double);
void   set_g_at_site(const int, const int, const double);
void   set_rho_u_at_site(const double, const double [], const int);
void   get_momentum_at_site(const int, double[ND]);

#endif
