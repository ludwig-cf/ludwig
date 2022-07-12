/****************************************************************************
 *
 *  fe_surfactant_oft_rt.h
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

#ifndef LUDWIG_SURFACTANT_OFT_RT_H
#define LUDWIG_SURFACTANT_OFT_RT_H

#include "pe.h"
#include "runtime.h"
#include "surfactant_oft.h"

__host__ int fe_surf_oft_param_rt(pe_t * pe, rt_t * rt, fe_surf_oft_param_t * p);
__host__ int fe_surf_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe,
				 field_t * phi);
__host__ int fe_surf_oft_psi_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe,
				 field_t * phi);
__host__ int fe_surf_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_surf_oft_t * fe,
						field_t * temperature);
#endif
