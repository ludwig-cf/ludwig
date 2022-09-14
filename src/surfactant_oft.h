/****************************************************************************
 *
 *  fe_surfactant_oft.h
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

#ifndef LUDWIG_FE_SURFACTANT_OFT_H
#define LUDWIG_FE_SURFACTANT_OFT_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_surfactant_oft_s fe_surf_oft_t;
typedef struct fe_surfactant_oft_param_s fe_surf_oft_param_t;

struct fe_surfactant_oft_param_s {
  double a;              /* Symmetric a */
  double b;              /* Symmetric b */
  double kappa;          /* Symmetric kappa */
  double kappa1;          /* Symmetric kappa1 */
  double kappa2;          /* Symmetric kappa2 */

  double kt;             /* Surfactant kT */
  double epsilon;        /* Surfactant epsilon */
  double beta;           /* Frumpkin isotherm */
  double w;              /* Surfactant w */
  
  double c;
  double h;
};

struct fe_surfactant_oft_s {
  fe_t super;                      /* "Superclass" block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  fe_surf_oft_param_t * param;         /* Parameters */
  field_t * phi;                   /* Single field with {phi,psi} */
  field_t * temperature;
  field_grad_t * dphi;             /* gradients thereof */
  fe_surf_oft_t * target;              /* Device copy */
};

__host__ int fe_surf_oft_create(pe_t * pe, cs_t * cs, field_t * phi,
			    field_grad_t * dphi, field_t * temperature, fe_surf_oft_param_t param,
			    fe_surf_oft_t ** fe);
__host__ int fe_surf_oft_free(fe_surf_oft_t * fe);
__host__ int fe_surf_oft_info(fe_surf_oft_t * fe);
__host__ int fe_surf_oft_param_set(fe_surf_oft_t * fe, fe_surf_oft_param_t vals);
__host__ int fe_surf_oft_sigma(fe_surf_oft_t * fe, double * sigma);
__host__ int fe_surf_oft_xi0(fe_surf_oft_t * fe,  double * xi0);
__host__ int fe_surf_oft_langmuir_isotherm(fe_surf_oft_t * fe, double * psi_c);
__host__ int fe_surf_oft_target(fe_surf_oft_t * fe, fe_t ** target);

__host__ int fe_surf_oft_param(fe_surf_oft_t * fe, fe_surf_oft_param_t * param);
__host__ int fe_surf_oft_fed(fe_surf_oft_t * fe, int index, double * fed);
__host__ int fe_surf_oft_mu(fe_surf_oft_t * fe, int index, double * mu);
__host__ int fe_surf_oft_str(fe_surf_oft_t * fe, int index, double s[3][3]);
__host__ int fe_surf_oft_str_v(fe_surf_oft_t * fe, int index, double s[3][3][NSIMDVL]);

#endif
