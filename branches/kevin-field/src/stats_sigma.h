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
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_SIGMA_H
#define STATS_SIGMA_H

#ifdef OLD_PHI
void stats_sigma_init(int nswitch);
void stats_sigma_measure(int ntime);
#else
#include "field.h"
int stats_sigma_init(field_t * phi, int nswtich);
int stats_sigma_measure(field_t * phi, int ntime);
#endif
#endif
