/*****************************************************************************
 *
 *  interaction.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef INTERACTION_H
#define INTERACTION_H

void      COLL_init(void);
void      COLL_update(void);

void      colloid_gravity(double f[3]);
void      colloid_gravity_set(const double f[3]);
double    colloid_forces_ahmax(void);

#endif
