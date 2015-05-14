/*****************************************************************************
 *
 *  gradients_rt.h
 *
 *  $Id: gradient_rt.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_RT_H
#define GRADIENT_RT_H

#include "field_grad.h"
#include "map.h"

__targetHost__ int gradient_rt_init(field_grad_t * grad, map_t * map);

#endif
