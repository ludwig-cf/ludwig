/****************************************************************************
 *
 *  brazovskii_rt.h
 *
 *  $Id: brazovskii_rt.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef BRAZOVSKII_RT_H
#define BRAZOVSKII_RT_H

#include "brazovskii.h"

__host__ int fe_brazovskii_run_time(fe_brazovskii_t * fe);
__host__ int fe_brazovskii_rt_init_phi(fe_brazovskii_t * fe, field_t * phi);

#endif
