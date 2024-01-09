/*****************************************************************************
 *
 *  polar_active.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_POLAR_ACTIVE_H
#define LUDWIG_POLAR_ACTIVE_H

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_polar_s fe_polar_t;
typedef struct fe_polar_param_s fe_polar_param_t;

/* Free energy parameters */

struct fe_polar_param_s {
  double a;                 /* Bulk parameter */
  double b;                 /* Bulk parameter */
  double kappa1;            /* Elastic constant */
  double delta;             /* Elastic constant */
  double kappa2;            /* Elastic constant */
  double zeta;              /* 'Activity' parameter */
  double lambda;            /* Flow aligning/tumbling parameter */
  double radius;            /* Used for spherical 'active region' */
};

/* Structure */

struct fe_polar_s {
  fe_t super;               /* Superclass */
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */
  fe_polar_param_t * param; /* Parameters */
  field_t * p;              /* Vector order parameter */
  field_grad_t * dp;        /* Gradients thereof */
  fe_polar_t * target;      /* Device pointer */
};


__host__ int fe_polar_create(pe_t * pe, cs_t * cs, field_t * p,
			     field_grad_t * dp, fe_polar_t ** fe);
__host__ int fe_polar_free(fe_polar_t * fe);
__host__ int fe_polar_param_set(fe_polar_t * fe, fe_polar_param_t values);
__host__ int fe_polar_param_commit(fe_polar_t * fe);
__host__ int fe_polar_param(fe_polar_t * fe, fe_polar_param_t * values);
__host__ int fe_polar_target(fe_polar_t * fe, fe_t ** target);

__host__ __device__ int fe_polar_fed(fe_polar_t * fe, int index, double * fed);
__host__ __device__ int fe_polar_mol_field(fe_polar_t * fe, int index, double h[3]);
__host__ __device__ int fe_polar_stress(fe_polar_t * fe, int index, double s[3][3]);
__host__ __device__ void fe_polar_stress_v(fe_polar_t * fe, int index, double s[3][3][NSIMDVL]);

#endif
