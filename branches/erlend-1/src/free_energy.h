/*****************************************************************************
 *
 *  free_energy
 *
 *  The free energy.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _FREE_ENERGY_H
#define _FREE_ENERGY_H

void init_free_energy(void);

double free_energy_A(void);
double free_energy_B(void);
double free_energy_K(void);
double surface_tension(void);
double interfacial_width(void);
double chemical_potential(const double, const double);
void   chemical_stress(double [3][3], const double, const double [],
		       const double);
double free_energy_density(const double, const double []);

#endif
