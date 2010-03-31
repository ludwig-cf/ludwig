/*****************************************************************************
 *
 *  gradient_2d_5pt_fluid.h
 *
 *  $Id: gradient_2d_5pt_fluid.h,v 1.1.2.3 2010-03-31 10:29:29 kevin Exp $
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

void gradient_2d_5pt_fluid_init(void);
void gradient_2d_5pt_fluid_d2(const int nop, const double * field,
			      double * grad, double * delsq);
void gradient_2d_5pt_fluid_d4(const int nop, const double * field,
			      double * grad, double * delsq);
void gradient_2d_5pt_fluid_dyadic(const int nop, const double * field,
				  double * grad, double * delsq);

#endif
