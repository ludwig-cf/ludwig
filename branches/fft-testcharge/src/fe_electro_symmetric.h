/*****************************************************************************
 *
 *  fe_electro_symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2013)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef FE_ELECTRO_SYMMETRIC_H
#define FE_ELECTRO_SYMMETRIC_H

#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "psi.h"

int fe_es_create(field_t * phi, field_grad_t * gradphi, psi_t * psi);
int fe_es_free(void);
int fe_es_mu_solv(int index, int n, double * mu);
int fe_es_deltamu_set(int nk, double * deltamu);
int fe_es_epsilon_set(double e1, double e2);
int fe_es_var_epsilon(int index, double * epsilon);

double fe_es_fed(const int index);
void fe_es_stress(const int index, double s[3][3]);

#endif
