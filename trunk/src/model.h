/*****************************************************************************
 *
 *  model.h
 *
 *  $Id: model.h,v 1.6 2006-10-12 14:09:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _MODEL_H
#define _MODEL_H

#ifdef _D3Q19_
  #include "d3q19.h"
#else
  #include "d3q15.h"
#endif

typedef struct {
  double f[NVEL], g[NVEL];
} Site;

extern const double rcs2;
extern const double d_[3][3];

void   model_init(void);
void   allocate_site(const int);

double get_eta_shear(void);
double get_eta_bulk(void);
double get_kT(void);
double get_rho0(void);
double get_phi0(void);

double get_f_at_site(const int, const int);
double get_g_at_site(const int, const int);
double get_rho_at_site(const int);
double get_phi_at_site(const int);
void   set_rho(const double, const int);
void   set_phi(const double, const int);
void   set_f_at_site(const int, const int, const double);
void   set_g_at_site(const int, const int, const double);
void   set_rho_u_at_site(const double, const double [], const int);

#endif
