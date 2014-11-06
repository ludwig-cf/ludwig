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
#include "field.h"
#include "field_grad.h"

#include "targetDP.h"

HOST int symmetric_phi_set(field_t * phi, field_grad_t * dphi);

HOST void   symmetric_free_energy_parameters_set(double a, double b, double kappa);
HOST double symmetric_a(void);
HOST double symmetric_b(void);
HOST double symmetric_interfacial_tension(void);
HOST double symmetric_interfacial_width(void);
HOST double symmetric_free_energy_density(const int index);
HOST double symmetric_chemical_potential(const int index, const int nop);
TARGET double symmetric_chemical_potential_target(const int index, const int nop, const double* t_phi, const double* t_delsqphi);
HOST double symmetric_isotropic_pressure(const int index);
HOST void   symmetric_chemical_stress(const int index, double s[3][3]);
TARGET void symmetric_chemical_stress_target(const int index, double s[3][3*NILP], const double* t_phi,  const double* t_gradphi, const double* t_delsqphi) {

#endif

