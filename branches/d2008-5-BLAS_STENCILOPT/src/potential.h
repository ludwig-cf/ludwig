/*****************************************************************************
 *
 *  potential.h
 *
 *  $Id: potential.h,v 1.4 2008-03-03 09:39:12 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _POTENTIAL_H
#define _POTENTIAL_H

extern const double ENERGY_HARD_SPHERE;

void soft_sphere_init(void);
void leonard_jones_init(void);
void yukawa_init(void);

double soft_sphere_energy(const double);
double soft_sphere_force(const double);
double leonard_jones_energy(const double);
double hard_sphere_energy(const double);
double yukawa_potential(const double);
double yukawa_force(const double);
double get_max_potential_range(void);

#endif
