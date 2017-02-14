/*****************************************************************************
 *
 *  blue_phase_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinbrugh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_RT_H
#define BLUE_PHASE_RT_H

#include "pe.h"
#include "runtime.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"

__host__ int blue_phase_init_rt(pe_t * pe, rt_t * rt,
				 fe_lc_t * fe,
				 beris_edw_t * be);
__host__ int blue_phase_rt_initial_conditions(pe_t * pe, rt_t * rt,
					      fe_lc_t * fe, field_t * q);

#endif
