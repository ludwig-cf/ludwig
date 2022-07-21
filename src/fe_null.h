/*****************************************************************************
 *
 *  fe_null.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FE_NULL_H
#define LUDWIG_FE_NULL_H

#include "pe.h"
#include "free_energy.h"

typedef struct fe_null_s fe_null_t;

struct fe_null_s {
  fe_t super;                  /* Superclass block */
  pe_t * pe;                   /* Parallel environment */
  fe_null_t * target;          /* Target copy */
};

__host__ int fe_null_create(pe_t * pe, fe_null_t ** fe);
__host__ int fe_null_free(fe_null_t * fe);
__host__ int fe_null_target(fe_null_t * fe, fe_t ** target);

__host__ __device__ int fe_null_fed(fe_null_t * fe, int index, double * fed);
__host__ __device__ int fe_null_mu(fe_null_t * fe, int index, double * mu);
__host__ __device__ int fe_null_str(fe_null_t * fe, int index, double s[3][3]);
__host__ __device__ void fe_null_str_v(fe_null_t * fe, int index,
				       double s[3][3][NSIMDVL]);

#endif
