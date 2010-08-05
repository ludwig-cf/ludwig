/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id: collision.h,v 1.4.14.4 2010-08-05 17:22:27 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLISION_H
#define COLLISION_H

void collide(void);
void test_isothermal_fluctuations(void);

void collision_ghost_modes_on(void);
void collision_ghost_modes_off(void);
void collision_fluctuations_on(void);
void collision_fluctuations_off(void);
void collision_relaxation_times_set(void);
void collision_relaxation_times(double * tau);

#endif
