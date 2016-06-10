/*****************************************************************************
 *
 *  stats_sigma.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_SIGMA_H
#define STATS_SIGMA_H

#include "symmetric.h"
#include "field.h"

int stats_sigma_init(fe_symm_t * fe, field_t * phi, int nswtich);
int stats_sigma_measure(field_t * phi, int ntime);

#endif
