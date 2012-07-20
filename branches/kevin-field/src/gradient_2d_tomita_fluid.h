/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_2D_TOMITA_FLUID_H
#define GRADIENT_2D_TOIMTA_FLUID_H

void gradient_2d_tomita_fluid_init(void);
void gradient_2d_tomita_fluid_d2(const int nop, const double * field,
				 double * grad, double * delsq);
void gradient_2d_tomita_fluid_d4(const int nop, const double * field,
				 double * grad, double * delsq);

#endif
