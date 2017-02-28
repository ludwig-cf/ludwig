/*****************************************************************************
 *
 *  advection_s.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_ADVECTION_S_H
#define LUDWIG_ADVECTION_S_H

#include "memory.h"
#include "leesedwards.h"
#include "advection.h"

struct advflux_s {

  int nf;        /* Number of fields (1 for scalar etc) */
  int nsite;     /* Number of sites allocated */
  double * fe;   /* East face flxues */
  double * fw;   /* West face flxues */
  double * fy;   /* y-face fluxes */
  double * fz;   /* z-face fluxes */

  pe_t * pe;           /* Parallel environment */
  cs_t * cs;           /* Coordinate system */
  lees_edw_t * le;     /* Lees Edwards */
  advflux_t * target;  /* copy of this structure on target */ 
};

#endif
