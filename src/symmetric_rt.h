/****************************************************************************
 *
 *  symmetric_rt.h
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

#ifndef SYMMETRIC_RT_H
#define SYMMETRIC_RT_H

#include "symmetric.h"

int fe_symmetric_run_time(field_t * phi, field_grad_t * dphi, fe_t ** fe);
int fe_symmetric_rt_initial_conditions(fe_symm_t * fe, field_t * phi);

#endif
