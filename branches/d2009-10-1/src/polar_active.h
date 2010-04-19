/*****************************************************************************
 *
 *  polar_active.h
 *
 *  $Id: polar_active.h,v 1.1.2.2 2010-04-19 10:32:36 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef POLAR_ACTIVE_H
#define POLAR_ACTIVE_H

/* This is an implementation of the abstract 'class' ... */ 

#include "free_energy_vector.h"

double polar_active_free_energy_density(const int index);
void   polar_active_chemical_stress(const int index, double s[3][3]);
void   polar_active_molecular_field(const int index, double h[3]);
void   polar_active_parameters_set(const double a, const double b,
				   const double kappa1,
				   const double kappa2);
void   polar_active_zeta_set(const double zeta);
double polar_active_zeta(void);
double polar_active_region(const int index);

#endif
