/*****************************************************************************
 *
 *  advection_s.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2019 The University of Edinburgh
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
  double * fe;   /* East face flxues (Lees-Edwards)   */
  double * fw;   /* West face flxues (Less-Edwards)   */
  double * fx;   /* x-face fluxes (between i and i+1) */
  double * fy;   /* y-face fluxes (between j and j+1) */
  double * fz;   /* z-face fluxes (between k and k+1) */

  pe_t * pe;           /* Parallel environment */
  cs_t * cs;           /* Coordinate system */
  lees_edw_t * le;     /* Lees Edwards */
  advflux_t * target;  /* copy of this structure on target */ 
};

#endif
