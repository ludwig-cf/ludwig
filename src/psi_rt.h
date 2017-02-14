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
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PSI_RT_H
#define PSI_RT_H

#include "pe.h"
#include "runtime.h"
#include "psi.h"
#include "map.h"

int psi_rt_init_param(pe_t * pe, rt_t * rt, psi_t * psi);
int psi_rt_init_rho(pe_t * pe, rt_t * rt, psi_t * psi, map_t * map);

#endif
