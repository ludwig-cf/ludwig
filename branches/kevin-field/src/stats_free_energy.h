/****************************************************************************
 *
 *  stats_free_energy.h
 *
 *  $Id: stats_free_energy.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef STATS_FREE_ENERGY_H
#define STATS_FREE_ENERGY_H

#ifdef OLD_PHI
void stats_free_energy_density(void);
void blue_phase_stats(int nstep);
#else

#include "field.h"
#include "field_grad.h"

int stats_free_energy_density(field_t * q);
int blue_phase_stats(field_t * q, field_grad_t * dq, int tstep);

#endif

#endif
