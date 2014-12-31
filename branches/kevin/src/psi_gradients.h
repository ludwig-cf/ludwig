/****************************************************************************
 *
 *  psi_gradients.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  (c) 2014 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef  PSI_GRADIENTS_H
#define  PSI_GRADIENTS_H

#include "map.h"
#include "psi.h"
#include "fe_electro_symmetric.h"

#ifdef NP_D3Q6
#define PSI_NGRAD 7
#endif

#ifdef NP_D3Q18
#define PSI_NGRAD 19
#endif

#ifdef NP_D3Q26
#define PSI_NGRAD 27
#endif

extern const int    psi_gr_cv[PSI_NGRAD][3];
extern const double psi_gr_wv[PSI_NGRAD];
extern const double psi_gr_rnorm[PSI_NGRAD];
extern const double psi_gr_rcs2;

int psi_electric_field(psi_t * psi, int index, double e[3]);
int psi_electric_field_d3qx(psi_t * psi, int index, double e[3]);
int psi_grad_rho_d3qx(psi_t * obj,  map_t * map, int index, int n,
		      double * grad_rho);
int psi_grad_eps_d3qx(psi_t * psi, f_vare_t fepsilon, int index,
		      double * grad_eps);

#endif                               
