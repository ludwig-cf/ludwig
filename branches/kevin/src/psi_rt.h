/*****************************************************************************
 *
 *  psi_rt.h
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

#ifndef PSI_RT_H
#define PSI_RT_H

#include "runtime.h"
#include "coords.h"
#include "psi.h"
#include "map.h"

int psi_init_param_rt(psi_t * psi, rt_t * rt);
int psi_init_rho_rt(psi_t * psi, coords_t * cs, map_t * map, rt_t * rt);

#endif
