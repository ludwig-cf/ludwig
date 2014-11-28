/*****************************************************************************
 *
 *  gradient_3d_7pt_solid.h
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

#ifndef GRADIENT_3D_7PT_SOLID_H
#define GRADIENT_3D_7PT_SOLID_H

#include "map.h"

int gradient_3d_7pt_solid_map_set(map_t * map);
int gradient_3d_7pt_solid_d2(const int nop, const double * field,double * t_field,
			     double * grad,double * t_grad, double * delsq, double * t_delsq, char * siteMask,char * t_siteMask);
#endif
