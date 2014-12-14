/*****************************************************************************
 *
 *  gradient_3d_27pt_fluid.h
 *
 *  $Id: gradient_3d_27pt_fluid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_3D_27PT_FLUID_H
#define GRADIENT_3D_27PT_FLUID_H

int gradient_3d_27pt_fluid_d2(const int nop, const double * field,double * t_field,
			      double * grad,double * t_grad, double * delsq, double * t_delsq);
int gradient_3d_27pt_fluid_d4(const int nop, const double * field,double * t_field,
			      double * grad,double * t_grad, double * delsq, double * t_delsq);

#endif
