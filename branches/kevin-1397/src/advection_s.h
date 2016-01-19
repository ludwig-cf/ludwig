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

#include "advection.h"

struct advflux_s {
  double * fe;   /* For LE planes */
  double * fw;   /* For LE planes */
  double * fy;
  double * fz;

  advflux_t * tcopy;  /* copy of this structure on target */ 
};

#ifndef OLD_SHIT

#include "memory.h"

#else

#ifdef LB_DATA_SOA
#define ADVADR ADDR_VECSITE_R
#else
#define ADVADR ADDR_VECSITE
#endif
#endif

#endif
