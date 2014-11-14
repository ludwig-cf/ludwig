/*****************************************************************************
 *
 *  free_energy.h
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

#include "targetDP.h"

typedef int (* f_mu_solv_t)(int index, int n, double * mu);

HOST int fe_create(void);

HOST void fe_density_set(double (* f)(const int index));
HOST void fe_chemical_potential_set(double (* f)(const int index, const int nop));
HOST void fe_isotropic_pressure_set(double (* f)(const int index));
HOST void fe_chemical_stress_set(void (* f)(const int index, double s[3][3]));

HOST double (* fe_density_function(void))(const int index);
HOST double (* fe_chemical_potential_function(void))(const int, const int nop);
HOST double (* fe_isotropic_pressure_function(void))(const int index);
HOST void   (* fe_chemical_stress_function(void))(const int index, double s[3][3]);

HOST int    fe_mu_solv_set(f_mu_solv_t function);
HOST int    fe_mu_solv(int index, int n, double * mu);
HOST double fe_kappa(void);
HOST void   fe_kappa_set(const double kappa);
HOST int    fe_set(void);

#endif
