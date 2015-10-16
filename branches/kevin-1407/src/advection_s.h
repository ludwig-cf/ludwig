/*****************************************************************************
 *
 *  advection_s.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
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

/* array of structures */
#define ADDR_ADV(nsite, nfield, index, ifield)	\
  ((nfield)*(index) + (ifield))


/* structure of arrays */
#define ADDR_ADV_R(nsite, nfield, index, ifield)	\
  ((nsite)*(ifield) + (index))

#ifdef LB_DATA_SOA
#define ADVADR ADDR_ADV_R
#else
#define ADVADR ADDR_ADV
#endif


#endif
