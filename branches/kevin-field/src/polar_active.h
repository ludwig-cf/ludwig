/*****************************************************************************
 *
 *  polar_active.h
 *
 *  $Id: polar_active.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
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
void   polar_active_region_radius_set(const double r);
double polar_active_region(const int index);

#ifdef OLD_PHI
#else
#include "field.h"
#include "field_grad.h"

int polar_active_p_set(field_t * p, field_grad_t * p_grad);
#endif

#endif
