/****************************************************************************
 *
 *  fe_two_symm_rt.h
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

#ifndef LUDWIG_TWO_BINARY_OFT_RT_H
#define LUDWIG_TWO_BINARY_OFT_RT_H

#include "pe.h"
#include "runtime.h"
#include "two_symm_oft.h"

__host__ int fe_two_symm_oft_param_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_param_t * p);
__host__ int fe_two_symm_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe,
				 field_t * phi);
__host__ int fe_two_symm_oft_psi_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe,
				 field_t * phi);
__host__ int fe_two_symm_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_two_symm_oft_t * fe,
						field_t * temperature);
int field_init_combine_insert(field_t * array, field_t * scalar, int nfin);

#endif
