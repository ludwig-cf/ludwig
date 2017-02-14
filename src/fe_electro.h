/*****************************************************************************
 *
 *  fe_electro.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FE_ELECTRO_H
#define FE_ELECTRO_H

#include "free_energy.h"
#include "psi.h"

typedef struct fe_electro_s fe_electro_t;

__host__ int fe_electro_create(psi_t * psi, fe_electro_t ** fe);
__host__ int fe_electro_free(fe_electro_t * fe);
__host__ int fe_electro_ext_set(fe_electro_t * fe, double ext_field[3]);
__host__ int fe_electro_target(fe_electro_t * fe, fe_t ** target);

__host__ int fe_electro_fed(fe_electro_t * fe, int index, double * fed);
__host__ int fe_electro_mu(fe_electro_t * fe, int index, double * mu);
__host__ int fe_electro_mu_solv(fe_electro_t * fe, int index, int k, double * mu);
__host__ int fe_electro_stress(fe_electro_t * fe,  int index, double s[3][3]);
__host__ int fe_electro_stress_ex(fe_electro_t * fe, int index, double s[3][3]);

#endif
