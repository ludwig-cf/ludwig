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
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_RT_H
#define SYMMETRIC_RT_H

#include "field.h"

void symmetric_run_time(void);
int symmetric_rt_initial_conditions(field_t * phi);

#endif
