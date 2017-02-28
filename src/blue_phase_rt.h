/*****************************************************************************
 *
 *  blue_phase_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2017 The University of Edinbrugh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_BLUE_PHASE_RT_H
#define LUDWIG_BLUE_PHASE_RT_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"

__host__ int blue_phase_init_rt(pe_t * pe, rt_t * rt,
				 fe_lc_t * fe,
				 beris_edw_t * be);
__host__ int blue_phase_rt_initial_conditions(pe_t * pe, rt_t * rt, cs_t * cs,
					      fe_lc_t * fe, field_t * q);

#endif
