/****************************************************************************
 *
 *  surfactant.h
 *
 *  $Id: surfactant.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef FE_SURFACTANT_H
#define FE_SURFACTANT_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_surfactant1_s fe_surfactant1_t;
typedef struct fe_surfactant1_param_s fe_surfactant1_param_t;

struct fe_surfactant1_param_s {
  double a;              /* Symmetric a */
  double b;              /* Symmetric b */
  double kappa;          /* Symmetric kappa */

  double kt;             /* Surfactant kT */
  double epsilon;        /* Surfactant epsilon */
  double beta;           /* Frumpkin isotherm */
  double w;              /* Surfactant w */
};

__host__ int fe_surfactant1_create(pe_t * pe, cs_t * cs, field_t * phi,
				   field_grad_t * dphi,
				   fe_surfactant1_t ** fe);
__host__ int fe_surfactant1_free(fe_surfactant1_t * fe);
__host__ int fe_surfactant1_param_set(fe_surfactant1_t * fe, fe_surfactant1_param_t vals);
__host__ int fe_surfactant1_sigma(fe_surfactant1_t * fe, double * sigma);
__host__ int fe_surfactant1_xi0(fe_surfactant1_t * fe,  double * xi0);
__host__ int fe_surfactant1_langmuir_isotherm(fe_surfactant1_t * fe, double * psi_c);
__host__ int fe_surfactant1_target(fe_surfactant1_t * fe, fe_t ** target);

__host__ int fe_surfactant1_param(fe_surfactant1_t * fe, fe_surfactant1_param_t * param);
__host__ int fe_surfactant1_fed(fe_surfactant1_t * fe, int index, double * fed);
__host__ int fe_surfactant1_mu(fe_surfactant1_t * fe, int index, double * mu);
__host__ int fe_surfactant1_str(fe_surfactant1_t * fe, int index, double s[3][3]);


#endif
