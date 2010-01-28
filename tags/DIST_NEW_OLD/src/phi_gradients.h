/*****************************************************************************
 *
 *  phi_gradients.h
 *
 *  $Id: phi_gradients.h,v 1.3 2009-05-29 06:57:56 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Phyiscs Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_GRADIENTS_H
#define PHI_GRADIENTS_H

void phi_gradients_compute(void);
void phi_gradients_set_fluid(void);
void phi_gradients_set_solid(void);
void phi_gradients_double_fluid(void); /* must be available to test */

#endif
