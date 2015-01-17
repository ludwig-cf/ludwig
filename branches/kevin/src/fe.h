/*****************************************************************************
 *
 *  fe.h
 *
 *  The 'abstract' free energy interface.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_H
#define FREE_ENERGY_H

#include "pe.h"

typedef struct fe_s fe_t;

typedef int (* fe_fed_ft)(fe_t * fe, int index, double * fed);
typedef int (* fe_mu_ft)(fe_t * fe, int index, double * mu);
typedef int (* fe_str_ft)(fe_t * fe, int index, double s[3][3]);
typedef int (* fe_mu_solv_ft)(fe_t * fe, int index, int n, double * mu);
typedef int (* fe_hvector_ft)(fe_t * fe, int index, double h[3]);
typedef int (* fe_htensor_ft)(fe_t * fe, int index, double h[3][3]);

__host__ int fe_create(fe_t ** p);
__host__ int fe_free(fe_t * fe);
__host__ int fe_child(fe_t * fe, void ** child);
__host__ int fe_register_cb(fe_t * fe, void * abstr, fe_fed_ft, fe_mu_ft,
			    fe_str_ft, fe_mu_solv_ft, fe_hvector_ft,
			    fe_htensor_ft);

__host__ int fe_fed(fe_t * fe, int index, double * fed);
__host__ int fe_mu(fe_t * fe, int index, double * mu);
__host__ int fe_str(fe_t * fe, int index, double s[3][3]);
__host__ int fe_mu_solv(fe_t * fe, int index, int n, double * mu); 
__host__ int fe_hvector(fe_t * fe, int index, double h[3]);
#endif
