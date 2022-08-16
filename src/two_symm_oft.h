/****************************************************************************
 *
 *  two_symm_oft.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_TWO_BINARY_OFT_H
#define LUDWIG_FE_TWO_BINARY_OFT_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_two_symm_oft_s fe_two_symm_oft_t;
typedef struct fe_two_symm_oft_param_s fe_two_symm_oft_param_t;

struct fe_two_symm_oft_param_s {
  double phi_a;              /* Symmetric phia */
  double phi_b;              /* Symmetric b */
  double phi_kappa0;         /* Symmetric kappa */
  double phi_kappa1;         /* Symmetric kappa1 */

  double psi_a;              /* Symmetric a */
  double psi_b;              /* Symmetric b */
  double psi_kappa;         /* Symmetric kappa */
  
  double c;
  double h;
};

struct fe_two_symm_oft_s {
  fe_t super;                      /* "Superclass" block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  fe_two_symm_oft_param_t * param;         /* Parameters */
  field_t * phi;                   /* Single field with {phi,psi} */
  field_t * temperature;
  field_grad_t * dphi;             /* gradients thereof */
  fe_two_symm_oft_t * target;              /* Device copy */
};

__host__ int fe_two_symm_oft_create(pe_t * pe, cs_t * cs, field_t * phi,
			    field_grad_t * dphi, field_t * temperature, fe_two_symm_oft_param_t param,
			    fe_two_symm_oft_t ** fe);
__host__ int fe_two_symm_oft_free(fe_two_symm_oft_t * fe);
__host__ int fe_two_symm_oft_info(fe_two_symm_oft_t * fe);
__host__ int fe_two_symm_oft_param_set(fe_two_symm_oft_t * fe, fe_two_symm_oft_param_t vals);
__host__ int fe_two_symm_oft_sigma(fe_two_symm_oft_t * fe, double * phi_sigma, double * psi_sigma);
__host__ int fe_two_symm_oft_xi0(fe_two_symm_oft_t * fe,  double * phi_xi, double * psi_xi);
__host__ int fe_two_symm_oft_target(fe_two_symm_oft_t * fe, fe_t ** target);

__host__ int fe_two_symm_oft_param(fe_two_symm_oft_t * fe, fe_two_symm_oft_param_t * param);
__host__ int fe_two_symm_oft_fed(fe_two_symm_oft_t * fe, int index, double * fed);
__host__ int fe_two_symm_oft_mu(fe_two_symm_oft_t * fe, int index, double * mu);
__host__ int fe_two_symm_oft_str(fe_two_symm_oft_t * fe, int index, double s[3][3]);
__host__ int fe_two_symm_oft_str_v(fe_two_symm_oft_t * fe, int index, double s[3][3][NSIMDVL]);

#endif
