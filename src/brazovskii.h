/****************************************************************************
 *
 *  brazovskii.h
 *
 *  $Id: brazovskii.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
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

#ifndef FE_BRAZOVSKII_H
#define FE_BRAZOVSKII_H

#include "pe.h"
#include "coords.h"
#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_brazovskii_s fe_brazovskii_t;
typedef struct fe_brazovskii_param_s fe_brazovskii_param_t;

struct fe_brazovskii_param_s {
  double a;
  double b;
  double c;
  double kappa;
};

__host__ int fe_brazovskii_create(pe_t * pe, cs_t * cs, field_t * phi,
				  field_grad_t * dphi,
				  fe_brazovskii_t ** p);
__host__ int fe_brazovskii_free(fe_brazovskii_t * fe);
__host__ int fe_brazovskii_param_set(fe_brazovskii_t * fe,
				     fe_brazovskii_param_t param);
__host__ int fe_brazovskii_target(fe_brazovskii_t * fe, fe_t ** target);

/* Host / target functions */

__host__ __device__ int fe_brazovskii_param(fe_brazovskii_t * fe,
					    fe_brazovskii_param_t * param);
__host__ __device__ int fe_brazovskii_amplitude(fe_brazovskii_t * fe,
						double * a0);
__host__ __device__ int fe_brazovskii_wavelength(fe_brazovskii_t * fe,
						 double * lamb);
__host__ __device__ int fe_brazovskii_fed(fe_brazovskii_t * fe, int index,
					  double * fed);
__host__ __device__ int fe_brazovskii_mu(fe_brazovskii_t * fe, int index,
					 double * mu);
__host__ __device__ int fe_brazovskii_str(fe_brazovskii_t * fe, int index,
					  double s[3][3]);
__host__ __device__ void fe_brazovskii_str_v(fe_brazovskii_t * fe, int index,
					     double s[3][3][NSIMDVL]);

#endif
