/****************************************************************************
 *
 *  surfactant_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SURFACTANT_RT_H
#define SURFACTANT_RT_H

#include "surfactant.h"

__host__ int fe_surfactant1_run_time(field_t * phi, field_grad_t * dphi,
				     fe_surfactant1_t ** fe);

#endif
