/*****************************************************************************
 *
 *  gradient_3d_27pt_solid.h
 *
 *  $Id: gradient_3d_27pt_solid.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_3D_27PT_SOLID_H
#define GRADIENT_3D_27PT_SOLID_H

int gradient_3d_27pt_solid_d2(const int nop, const double * field,
			      double * grad, double * delsq);

#endif
