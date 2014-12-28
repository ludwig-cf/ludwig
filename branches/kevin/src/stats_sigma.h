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

typedef struct stats_sigma_s stats_sigma_t;

#include "coords.h"
#include "field.h"

int stats_sigma_create(coords_t * cs, field_t * phi, stats_sigma_t ** stat);
int stats_sigma_free(stats_sigma_t * stat);
int stats_sigma_free(stats_sigma_t * stat);
int stats_sigma_measure(stats_sigma_t * stat, field_t * phi, int ntime);

#endif
