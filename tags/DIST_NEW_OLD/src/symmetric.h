/****************************************************************************
 *
 *  symmetric.h
 *
 *  $Id: symmetric.h,v 1.1.2.1 2009-11-04 09:52:12 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

void   symmetric_free_energy_parameters_set(double a, double b, double kappa);
double symmetric_interfacial_tension(void);
double symmetric_interfacial_width(void);
double symmetric_free_energy_density(const int index);
double symmetric_chemical_potential(const int index, const int nop);
double symmetric_isotropic_pressure(const int index);
void   symmetric_chemical_stress(const int index, double s[3][3]);

#endif

