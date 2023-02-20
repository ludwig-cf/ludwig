/*****************************************************************************
 *
 *  psi_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_RT_H
#define LUDWIG_PSI_RT_H

#include "pe.h"
#include "runtime.h"
#include "psi.h"
#include "map.h"

int psi_rt_init_rho(pe_t * pe, rt_t * rt, psi_t * psi, map_t * map);

int psi_options_rt(pe_t * pe, cs_t * cs, rt_t * rt, psi_options_t * opts);
int psi_info(pe_t * pe, const psi_t * psi);

#endif
