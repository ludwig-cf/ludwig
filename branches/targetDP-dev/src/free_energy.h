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

typedef int (* f_mu_solv_t)(int index, int n, double * mu);

int fe_create(void);

void fe_density_set(double (* f)(const int index));
void fe_chemical_potential_set(double (* f)(const int index, const int nop));
void fe_isotropic_pressure_set(double (* f)(const int index));
void fe_chemical_stress_set(void (* f)(const int index, double s[3][3]));

double (* fe_density_function(void))(const int index);
double (* fe_chemical_potential_function(void))(const int, const int nop);
double (* fe_isotropic_pressure_function(void))(const int index);
void   (* fe_chemical_stress_function(void))(const int index, double s[3][3]);

int    fe_mu_solv_set(f_mu_solv_t function);
int    fe_mu_solv(int index, int n, double * mu);
double fe_kappa(void);
void   fe_kappa_set(const double kappa);
int    fe_set(void);

#endif
