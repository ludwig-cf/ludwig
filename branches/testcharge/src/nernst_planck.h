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
#include "colloids.h"

int nernst_planck_driver(psi_t * psi, hydro_t * hydro, map_t * map);
int nernst_planck_driver_d3q18(psi_t * psi, hydro_t * hydro, map_t * map, colloids_info_t * cinfo);
int nernst_planck_adjust_multistep(psi_t * psi);

int nernst_planck_maxacc(double * acc);
int nernst_planck_maxacc_set(double acc);
#endif
