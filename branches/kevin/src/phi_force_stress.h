/*****************************************************************************
 *
 *  phi_force_stress.h
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2012)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PHI_FORCE_STRESS_H
#define PHI_FORCE_STRESS_H

#include "coords.h"

int phi_force_stress_allocate(coords_t * cs, double ** p);
int phi_force_stress_free(double * p);
int phi_force_stress(double * p3d, int index, double p[3][3]);
int phi_force_stress_set(double * p3d, int index, double p[3][3]);
int phi_force_stress_compute(coords_t * cs, double * p3d);

#endif
