/*****************************************************************************
 *
 *  advection_s.h
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
  le_t * le;     /* Reference to Lees Edwards information */
  int nf;        /* Number of fields */
  int nsites;    /* Number of sites */
  double * fe;   /* East face fluxes */
  double * fw;   /* West faces fluxes */
  double * fy;   /* Unique y-fluxes */
  double * fz;   /* Unique z-fluxes */
};

#endif
