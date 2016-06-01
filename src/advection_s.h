/*****************************************************************************
 *
 *  advection_s.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef ADVECTION_S_H
#define ADVECTION_S_H

#include "memory.h"
#include "advection.h"

struct advflux_s {
  double * fe;   /* For LE planes */
  double * fw;   /* For LE planes */
  double * fy;
  double * fz;

  advflux_t * tcopy;  /* copy of this structure on target */ 
};

#endif
