/*****************************************************************************
 *
 *  fe_electro.h
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

#ifndef FE_ELECTRO_H
#define FE_ELECTRO_H

#include "free_energy.h"
#include "psi.h"

int fe_electro_create(psi_t * psi);
int fe_electro_free(void);
int fe_electro_ext_set(double ext_field[3]);

double fe_electro_fed(const int index);
double fe_electro_mu(const int index, const int n);
void fe_electro_stress(const int index, double s[3][3]);

#endif
