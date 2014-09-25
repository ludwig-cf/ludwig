/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid.h
 *
 *  $Id: gradient_2d_5pt_fluid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_2D_5PT_FLUID_H
#define GRADIENT_2D_5PT_FLUID_H

int gradient_2d_5pt_fluid_d2(const int nop, const double * field,
			     double * grad, double * delsq);
int gradient_2d_5pt_fluid_d4(const int nop, const double * field,
			     double * grad, double * delsq);

#endif
