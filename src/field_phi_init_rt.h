/*****************************************************************************
 *
 *  field_phi_init_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_PHI_INIT_RT_H
#define LUDWIG_FIELD_PHI_INIT_RT_H

#include "pe.h"
#include "runtime.h"
#include "field_phi_init.h"

int field_phi_init_rt(pe_t * pe, rt_t * rt, field_phi_info_t param,
		      field_t * phi);

#endif
