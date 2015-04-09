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

__targetHost__  void phi_force_stress(const int index, double p[3][3]);
__targetHost__  void phi_force_stress_set(const int index, double p[3][3]);
__targetHost__  void phi_force_stress_compute(field_t* q, field_grad_t* q_grad);
__targetHost__  void phi_force_stress_allocate(void);
__targetHost__  void phi_force_stress_free(void);

#endif
