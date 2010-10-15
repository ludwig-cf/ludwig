/*****************************************************************************
 *
 *  potential.h
 *
 *  $Id: potential.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statisitcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef POTENTIAL_H
#define POTENTIAL_H

extern const double ENERGY_HARD_SPHERE;

void soft_sphere_init(void);
void lennard_jones_init(void);
void yukawa_init(void);

double soft_sphere_energy(const double);
double soft_sphere_force(const double);
double lennard_jones_energy(const double);
double hard_sphere_energy(const double);
double yukawa_potential(const double);
double yukawa_force(const double);
double get_max_potential_range(void);

#endif
