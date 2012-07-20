/****************************************************************************
 *
 *  brazovskii.h
 *
 *  $Id: brazovskii.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#ifndef BRAZOVSKII_H
#define BRAZOVSKII_H

void   brazovskii_free_energy_parameters_set(double a, double b, double kappa,
					     double c);
double brazovskii_amplitude(void);
double brazovskii_wavelength(void);
double brazovskii_free_energy_density(const int index);
double brazovskii_chemical_potential(const int index, const int nop);
double brazovskii_isotropic_pressure(const int index);
void   brazovskii_chemical_stress(const int index, double s[3][3]);

#ifdef OLD_PHI
#else
#include "field.h"
#include "field_grad.h"

int brazovskii_phi_set(field_t * phi, field_grad_t * phi_grad);

#endif

#endif

