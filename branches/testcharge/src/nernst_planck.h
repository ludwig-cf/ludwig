/*****************************************************************************
 *
 *  nernst_planck.h
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

#ifndef NERNST_PLANCK_H
#define NERNST_PLANCK_H

#include "psi.h"
#include "hydro.h"
#include "map.h"

int nernst_planck_driver(psi_t * psi, hydro_t * hydro, map_t * map);

#endif
