/*****************************************************************************
 *
 *  fe.h
 *
 *  The 'abstract' free energy interface. See fe.c for a desription.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_H
#define FREE_ENERGY_H

#include "pe.h"

typedef struct fe_s fe_t;

__host__ int fe_free(fe_t * fe);
__host__ __device__ int fe_fed(fe_t * fe, int index, double * fed);
__host__ __device__ int fe_mu(fe_t * fe, int index, double * mu);
__host__ __device__ int fe_str(fe_t * fe, int index, double s[3][3]);
__host__ __device__ int fe_mu_solv(fe_t * fe, int index, int n, double * mu); 
__host__ __device__ int fe_hvector(fe_t * fe, int index, double h[3]);
__host__ __device__ int fe_htensor(fe_t * fe, int index, double d[3][3]);

#endif
