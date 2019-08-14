/****************************************************************************
 *
 *  psi_gradients.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  (c) 2014-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef  LUDWIG_PSI_GRADIENTS_H
#define  LUDWIG_PSI_GRADIENTS_H

#include "map.h"
#include "psi.h"
#include "fe_electro_symmetric.h"

#if defined NP_D3Q18

#define PSI_NGRAD 19

#elif defined NP_D3Q26

#define PSI_NGRAD 27

#else

/* Default to 7-point stencil */

#define PSI_NGRAD 7

#endif

extern const int    psi_gr_cv[PSI_NGRAD][3];
extern const double psi_gr_wv[PSI_NGRAD];
extern const double psi_gr_rnorm[PSI_NGRAD];
extern const double psi_gr_rcs2;

int psi_electric_field(psi_t * psi, int index, double e[3]);
int psi_electric_field_d3qx(psi_t * psi, int index, double e[3]);
int psi_grad_rho_d3qx(psi_t * obj,  map_t * map, int index, int n, double * grad_rho);
int psi_grad_eps_d3qx(psi_t * psi, fe_t * fe, f_vare_t fepsilon, int index,
		      double * grad_eps);

#endif                     
