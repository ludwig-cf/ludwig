/*****************************************************************************
 *
 *  fe_polar.h
 *
 *  $Id: polar_active.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FE_POLAR_H
#define FE_POLAR_H

#include "pe.h"
#include "fe.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_polar_s fe_polar_t;
typedef struct fe_polar_param_s fe_polar_param_t;

struct fe_polar_param_s {
  double a;
  double b;
  double delta;
  double kappa1;
  double kappa2;
  double lambda;
  double zeta;
};

__host__ int fe_polar_create(field_t * p, field_grad_t * dp,
			     fe_polar_t ** pobj);
__host__ int fe_polar_free(fe_polar_t * fe);
__host__ int fe_polar_param_set(fe_polar_t * fe, fe_polar_param_t values);

__host__ __device__ int fe_polar_param(fe_polar_t * fe,
				       fe_polar_param_t * values);
__host__ __device__ int fe_polar_fed(fe_polar_t * fe, int index, double * fed);
__host__ __device__ int fe_polar_mol_field(fe_polar_t * fe, int index,
					   double h[3]);
__host__ __device__ int fe_polar_stress(fe_polar_t * fe, int index,
					double s[3][3]);

#endif
