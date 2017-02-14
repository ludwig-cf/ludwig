/*****************************************************************************
 *
 *  coords_rt.h
 *
 *  $Id: coords_rt.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2016 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef COORDS_RT_H
#define COORDS_RT_H

#include "pe.h"
#include "runtime.h"
#include "coords.h"

int coords_init_rt(pe_t * pe, rt_t * rt, cs_t * cs);

#endif
