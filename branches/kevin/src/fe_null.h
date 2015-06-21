/*****************************************************************************
 *
 *  fe_null.h
 *
 *  A 'null' free energy used if single fluid only.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FE_NULL_H
#define FE_NULL_H

#include "pe.h"
#include "fe.h"

typedef struct fe_null_s fe_null_t;

__host__ int fe_null_create(fe_null_t ** fe);
__host__ int fe_null_free(fe_null_t * fe);

__host__ __device__ int fe_null_fed(fe_null_t * fe, int index, double * fed);
__host__ __device__ int fe_null_mu(fe_null_t * fe, int index, double * mu);
__host__ __device__ int fe_null_str(fe_null_t * fe, int index, double s[3][3]);
__host__ __device__ int fe_null_mu_solv(fe_null_t * fe, int index, int n, double * mu); 
__host__ __device__ int fe_null_hvector(fe_null_t * fe, int index, double h[3]);
__host__ __device__ int fe_null_htensor(fe_null_t * fe, int index, double d[3][3]);

#endif
