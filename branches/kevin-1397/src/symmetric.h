/****************************************************************************
 *
 *  symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_symm_param_s fe_symm_param_t;
typedef struct fe_symm_s fe_symm_t;

/* Free energy structure */

struct fe_symm_s {
  fe_t super;
  fe_symm_param_t * param;     /* Parameters */
  field_t * phi;               /* Scalar order parameter or composition */
  field_grad_t * dphi;         /* Gradients thereof */
  fe_symm_t * target;          /* Target copy */
};

/* Parameters */

struct fe_symm_param_s {
  double a;
  double b;
  double kappa;
};

__host__ int fe_symm_create(field_t * f, field_grad_t * grd, fe_symm_t ** p);
__host__ int fe_symm_free(fe_symm_t * fe);
__host__ int fe_symm_param_set(fe_symm_t * fe, fe_symm_param_t values);
__host__ int fe_symm_target(fe_symm_t * fe, fe_t ** target);

__host__ __device__ int fe_symm_param(fe_symm_t * fe, fe_symm_param_t * values);
__host__ __device__ int fe_symm_interfacial_tension(fe_symm_t * fe, double * s);
__host__ __device__ int fe_symm_interfacial_width(fe_symm_t * fe, double * xi);
__host__ __device__ int fe_symm_fed(fe_symm_t * fe, int index, double * fed);
__host__ __device__ int fe_symm_mu(fe_symm_t * fe, int index, double * mu);
__host__ __device__ int fe_symm_str(fe_symm_t * fe, int index, double s[3][3]);

__target__
void fe_symm_chemical_stress_target(fe_symm_t * fe, int index,
						 double s[3][3*NSIMDVL]);

#endif

