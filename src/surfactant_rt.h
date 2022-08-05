/****************************************************************************
 *
 *  fe_surfactant1_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_SURFACTANT1_RT_H
#define LUDWIG_SURFACTANT1_RT_H

#include "pe.h"
#include "runtime.h"
#include "surfactant.h"

__host__ int fe_surf_param_rt(pe_t * pe, rt_t * rt, fe_surf_param_t * p);
__host__ int fe_surf_phi_init_rt(pe_t * pe, rt_t * rt, fe_surf_t * fe,
				 field_t * phi);
__host__ int fe_surf_psi_init_rt(pe_t * pe, rt_t * rt, fe_surf_t * fe,
				 field_t * phi);
int field_init_combine_insert(field_t * array, field_t * scalar, int nfin);
#endif
