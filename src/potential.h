/*****************************************************************************
 *
 *  potential.h
 *
 *  $Id: potential.h,v 1.1 2006-10-18 17:47:02 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _POTENTIAL_H
#define _POTENTIAL_H

extern const double ENERGY_HARD_SPHERE;

void soft_sphere_init(void);

double soft_sphere_energy(const double);
double soft_sphere_force(const double);
double hard_sphere_energy(const double);
double get_max_potential_range(void);

#endif
