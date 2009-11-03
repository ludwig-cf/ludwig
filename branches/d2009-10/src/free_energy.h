/*****************************************************************************
 *
 *  free_energy
 *
 *  The free energy.
 *
 *  $Id: free_energy.h,v 1.4 2008-11-14 14:42:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
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
double free_energy_density(const int);
double free_energy_chemical_potential(const int, const int);
double free_energy_get_chemical_potential(const int);
double free_energy_get_isotropic_pressure(const int);
void   free_energy_get_chemical_stress(const int, double [3][3]);

void free_energy_set_A(const double);
void free_energy_set_B(const double);
void free_energy_set_kappa(const double);
int  free_energy_is_brazovskii(void);

#endif
