/****************************************************************************
 *
 *  symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

#include "free_energy.h"

void   symmetric_free_energy_parameters_set(double a, double b, double kappa);
double symmetric_a(void);
double symmetric_b(void);
double symmetric_interfacial_tension(void);
double symmetric_interfacial_width(void);
double symmetric_free_energy_density(const int index);
double symmetric_chemical_potential(const int index, const int nop);
double symmetric_isotropic_pressure(const int index);
void   symmetric_chemical_stress(const int index, double s[3][3]);

#ifdef OLD_PHI
#else
#include "field.h"
#include "field_grad.h"
int symmetric_phi_set(field_t * phi, field_grad_t * dphi);
#endif

#endif

