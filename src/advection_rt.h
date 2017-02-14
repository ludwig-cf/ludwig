/*****************************************************************************
 *
 *  advection_rt.h
 *
 *  $Id: advection_rt.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_RT_H
#define ADVECTION_RT_H

#include "pe.h"
#include "runtime.h"

int advection_init_rt(pe_t * pe, rt_t * rt);

#endif
