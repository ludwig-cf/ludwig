/*****************************************************************************
 *
 *  phi_bc_open_rt.h
 *
 *  Composition open boundaries. See phi_bc_open_rt.c for details.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_BC_OPEN_RT_H
#define LUDWIG_PHI_BC_OPEN_RT_H

#include "pe.h"
#include "coords.h"
#include "phi_bc_open.h"
#include "runtime.h"

__host__ int phi_bc_open_rt(pe_t * pe, rt_t * rt, cs_t * cs,
			    phi_bc_open_t ** inflow,
			    phi_bc_open_t ** outflow);
#endif
