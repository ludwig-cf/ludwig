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
 *  (c) 2009-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_H
#define FREE_ENERGY_H

#include "targetDP.h"

typedef void fe_t;

typedef int (* f_mu_solv_t)(int index, int n, double * mu);

__targetHost__ int fe_create(void);

__targetHost__ void fe_density_set(double (* f)(const int index));
__targetHost__ void fe_chemical_potential_set(double (* f)(const int index, const int nop));
__targetHost__ void fe_isotropic_pressure_set(double (* f)(const int index));
__targetHost__ void fe_chemical_stress_set(void (* f)(const int index, double s[3][3]));

__targetHost__ double (* fe_density_function(void))(const int index);
__targetHost__ double (* fe_chemical_potential_function(void))(const int, const int nop);
__targetHost__ double (* fe_isotropic_pressure_function(void))(const int index);
__targetHost__ void   (* fe_chemical_stress_function(void))(const int index, double s[3][3]);

__targetHost__ int    fe_mu_solv_set(f_mu_solv_t function);
__targetHost__ int    fe_mu_solv(int index, int n, double * mu);
__targetHost__ double fe_kappa(void);
__targetHost__ void   fe_kappa_set(const double kappa);
__targetHost__ int    fe_set(void);

#endif
