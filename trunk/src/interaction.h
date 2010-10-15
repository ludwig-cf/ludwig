/*****************************************************************************
 *
 *  interaction.h
 *
 *  $Id: interaction.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef INTERACTION_H
#define INTERACTION_H

void      COLL_zero_forces(void);
void      COLL_init(void);

double    COLL_interactions(void);
void      COLL_update(void);
void      COLL_forces(void);

double    soft_sphere_energy(const double);
double    soft_sphere_force(const double);

void      colloid_gravity(double f[3]);

#endif
