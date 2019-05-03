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

__host__ int fe_surf1_param_rt(pe_t * pe, rt_t * rt, fe_surf1_param_t * p);

#endif
