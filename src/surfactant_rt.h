/****************************************************************************
 *
 *  surfactant_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef SURFACTANT_RT_H
#define SURFACTANT_RT_H

#include "runtime.h"
#include "surfactant.h"

__host__ int fe_surfactant1_run_time(pe_t * pe, cs_t * cs, rt_t * rt,
				     field_t * phi, field_grad_t * dphi,
				     fe_surfactant1_t ** fe);

#endif
