/****************************************************************************
 *
 *  fe_surfactant.h
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

#ifndef LUDWIG_FE_SURFACTANT1_H
#define LUDWIG_FE_SURFACTANT1_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_surfactant1_s fe_surf1_t;
typedef struct fe_surfactant1_param_s fe_surf1_param_t;

struct fe_surfactant1_param_s {
  double a;              /* Symmetric a */
  double b;              /* Symmetric b */
  double kappa;          /* Symmetric kappa */

  double kt;             /* Surfactant kT */
  double epsilon;        /* Surfactant epsilon */
  double beta;           /* Frumpkin isotherm */
  double w;              /* Surfactant w */
};

__host__ int fe_surf1_create(pe_t * pe, cs_t * cs, field_t * phi,
				   field_grad_t * dphi, fe_surf1_param_t param,
				   fe_surf1_t ** fe);
__host__ int fe_surf1_free(fe_surf1_t * fe);
__host__ int fe_surf1_info(fe_surf1_t * fe);
__host__ int fe_surf1_param_set(fe_surf1_t * fe, fe_surf1_param_t vals);
__host__ int fe_surf1_sigma(fe_surf1_t * fe, double * sigma);
__host__ int fe_surf1_xi0(fe_surf1_t * fe,  double * xi0);
__host__ int fe_surf1_langmuir_isotherm(fe_surf1_t * fe, double * psi_c);
__host__ int fe_surf1_target(fe_surf1_t * fe, fe_t ** target);

__host__ int fe_surf1_param(fe_surf1_t * fe, fe_surf1_param_t * param);
__host__ int fe_surf1_fed(fe_surf1_t * fe, int index, double * fed);
__host__ int fe_surf1_mu(fe_surf1_t * fe, int index, double * mu);
__host__ int fe_surf1_str(fe_surf1_t * fe, int index, double s[3][3]);
__host__ int fe_surf1_str_v(fe_surf1_t * fe, int index, double s[3][3][NSIMDVL]);

#endif
