/****************************************************************************
 *
 *  fe_symmetric.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

#include "pe.h"
#include "fe.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_symmetric_param_s fe_symmetric_param_t;
typedef struct fe_symmetric_s fe_symmetric_t;

struct fe_symmetric_param_s {
  double a;
  double b;
  double kappa;
};

__host__ int fe_symmetric_create(fe_t * fe, field_t * f, field_grad_t * grd,
				 fe_symmetric_t ** p);
__host__ int fe_symmetric_free(fe_symmetric_t * fe);
__host__ int fe_symmetric_param_set(fe_symmetric_t * fe,
				    fe_symmetric_param_t values);


/* Host / target functions */
__host__ __device__ int fe_symmetric_param(fe_symmetric_t * fe,
					   fe_symmetric_param_t * values);
__host__ __device__ int fe_symmetric_interfacial_tension(fe_symmetric_t * fe,
							 double * s);
__host__ __device__ int fe_symmetric_interfacial_width(fe_symmetric_t * fe,
						       double * xi);
__host__ __device__ int fe_symmetric_fed(fe_symmetric_t * fe, int index,
					 double * fed);
__host__ __device__ int fe_symmetric_mu(fe_symmetric_t * fe, int index,
					double * mu);
__host__ __device__ int fe_symmetric_str(fe_symmetric_t * fe, int index,
					 double s[3][3]);
#endif
