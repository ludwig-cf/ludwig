/****************************************************************************
 *
 *  fe_symmetric_ll.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_SYMMETRIC_LL_H
#define LUDWIG_FE_SYMMETRIC_LL_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_symmetric_ll_s fe_symmetric_ll_t;
typedef struct fe_symmetric_ll_param_s fe_symmetric_ll_param_t;

struct fe_symmetric_ll_s {
  fe_t super;                       /* "Superclass" block */
  pe_t * pe;                        /* Parallel environment */
  cs_t * cs;                        /* Coordinate system */
  fe_symmetric_ll_param_t * param;       /* Parameters */
  field_t * phi;                    /* Single field with {phi,psi} */
  field_grad_t * dphi;              /* gradients thereof */
  fe_symmetric_ll_t * target;            /* Device copy */
};

struct fe_symmetric_ll_param_s {
  double a1;
  double b1;
  double kappa1;

  double a2;
  double b2;
  double kappa2;
};


__host__ int fe_symmetric_ll_create(pe_t * pe, cs_t * cs, field_t * phi,
                             field_grad_t * dphi, fe_symmetric_ll_param_t param,
                             fe_symmetric_ll_t ** fe);
__host__ int fe_symmetric_ll_free(fe_symmetric_ll_t * fe);
__host__ int fe_symmetric_ll_param_set(fe_symmetric_ll_t * fe, fe_symmetric_ll_param_t vals);
__host__ int fe_symmetric_ll_target(fe_symmetric_ll_t * fe, fe_t ** target);

__host__ int fe_symmetric_ll_info(fe_symmetric_ll_t * fe);

__host__ int fe_symmetric_ll_sigma(fe_symmetric_ll_t * fe, double * sigma);
__host__ int fe_symmetric_ll_param(fe_symmetric_ll_t * fe, fe_symmetric_ll_param_t * param);

__host__ __device__ int fe_symmetric_ll_fed(fe_symmetric_ll_t * fe, int index,
				       double * fed);
__host__ __device__ int fe_symmetric_ll_mu(fe_symmetric_ll_t * fe, int index,
				      double * mu);
__host__ __device__ int fe_symmetric_ll_str(fe_symmetric_ll_t * fe, int index,
				       double s[3][3]);
__host__ __device__ int fe_symmetric_ll_str_v(fe_symmetric_ll_t * fe, int index,
					 double s[3][3][NSIMDVL]);

#endif
