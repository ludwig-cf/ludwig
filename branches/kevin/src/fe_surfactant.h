/****************************************************************************
 *
 *  fe_surfactant.h
 *
 *  $Id: surfactant.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef FE_SURFACTANT_H
#define FE_SURFACTANT_H

#include "pe.h"
#include "fe.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_surfactant_s fe_surfactant_t;
typedef struct fe_surfactant_param_s fe_surfactant_param_t;

struct fe_surfactant_param_s {
  double a;
  double b;
  double kappa;
  double kt;
  double epsilon;
  double beta;
  double w;
};

__host__ int fe_surfactant_create(field_t * phi, field_grad_t * dphi,
				  fe_surfactant_t ** p);
__host__ int fe_surfactant_free(fe_surfactant_t * fe);
__host__ int fe_surfactant_param_set(fe_surfactant_t * fe,
				     fe_surfactant_param_t values);

__host__ __device__ int fe_surfactant_param(fe_surfactant_t * fe,
					     fe_surfactant_param_t * values);
__host__ __device__ int fe_surfactant_sigma(fe_surfactant_t * fe,
					     double * sigma);
__host__ __device__ int fe_surfactant_xi0(fe_surfactant_t * fe,
					  double * xi0);
__host__ __device__ int fe_surfactant_langmuir_isotherm(fe_surfactant_t * fe,
							double * kt);
__host__ __device__ int fe_surfactant_fed(fe_surfactant_t * fe, int index,
					  double * fed);
__host__ __device__ int fe_surfactant_mu(fe_surfactant_t * fe, int index,
					 double * mu);
__host__ __device__ int fe_surfactant_str(fe_surfactant_t * fe, int index,
					  double s[3][3]);

#endif

