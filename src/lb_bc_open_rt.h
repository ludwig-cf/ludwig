/*****************************************************************************
 *
 *  lb_bc_open_rt.h
 *
 *  Open boundary condition run-time initialisation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_BC_OPEN_RT_H
#define LUDWIG_LB_BC_OPEN_RT_H

#include "pe.h"
#include "coords.h"
#include "lb_bc_open.h"
#include "runtime.h"

__host__ int lb_bc_open_rt(pe_t * pe, rt_t * rt, cs_t * cs, lb_t * lb,
			   lb_bc_open_t ** inflow,
			   lb_bc_open_t ** outflow);
#endif
