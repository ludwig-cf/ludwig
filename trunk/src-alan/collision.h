/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id$
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

enum relaxation {RELAXATION_M10, RELAXATION_BGK, RELAXATION_TRT};

void collide(void);
void test_isothermal_fluctuations(void);

void collision_ghost_modes_on(void);
void collision_ghost_modes_off(void);
void collision_fluctuations_on(void);
void collision_fluctuations_off(void);
void collision_relaxation_times_set(void);
void collision_relaxation_times(double * tau);
void collision_relaxation_set(const int nrelax);
void collision_init(void);
void collision_finish(void);

#endif
