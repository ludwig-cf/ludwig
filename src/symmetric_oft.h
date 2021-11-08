/****************************************************************************
 *
 *  symmetric_oft.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_TEMPERATURE_H
#define LUDWIG_TEMPERATURE_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
typedef struct fe_symm_oft_param_s fe_symm_oft_param_t;
typedef struct fe_symm_oft_s fe_symm_oft_t;

/* Free energy structure */

struct fe_symm_oft_s {
  fe_t super;
  pe_t * pe;                   /* Parallel environment */
  cs_t * cs;                   /* Coordinate system */
  io_options_t * options;
  fe_symm_oft_param_t * param;     /* Parameters */
  field_t * phi;               /* Scalar order parameter or composition */
  field_grad_t * dphi;         /* Gradients thereof */
  field_t * temperature;         /* Temperature */
  fe_symm_oft_t * target;          /* Target copy */
};

/* Parameters */

struct fe_symm_oft_param_s {
  double a; 
  double b;
  double kappa;
  double lambda;                    /* heat diffusivity */
  double entropy;
};

__host__ int fe_symm_oft_create(pe_t * pe, cs_t * cs, field_t * f,
			    field_grad_t * grd, field_t * temperature, io_options_t * options, fe_symm_oft_t ** p);
__host__ int fe_symm_oft_free(fe_symm_oft_t * fe);
__host__ int fe_symm_oft_param_set(fe_symm_oft_t * fe, fe_symm_oft_param_t values);
__host__ int fe_symm_oft_target(fe_symm_oft_t * fe, fe_t ** target);

__host__ __device__ int fe_symm_oft_param(fe_symm_oft_t * fe, fe_symm_oft_param_t * values);
__host__ __device__ int fe_symm_oft_fed(fe_symm_oft_t * fe, int index, double * fed);
__host__ __device__ int fe_symm_oft_mu(fe_symm_oft_t * fe, int index, double * mu);

__host__ __device__ int fe_symm_oft_str(fe_symm_oft_t * fe, int index, double s[3][3]);
__host__ __device__ void fe_symm_oft_str_v(fe_symm_oft_t * fe, int index,
				       double s[3][3][NSIMDVL]);

#endif

