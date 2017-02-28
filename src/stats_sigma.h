/*****************************************************************************
 *
 *  stats_sigma.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_SIGMA_H
#define LUDWIG_STATS_SIGMA_H

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "symmetric.h"

typedef struct stats_sigma_s stats_sigma_t;

int stats_sigma_create(pe_t * pe, cs_t * cs, fe_symm_t * fe, field_t * phi,
		       stats_sigma_t ** pobj);
int stats_sigma_free(stats_sigma_t * stat);
int stats_sigma_measure(stats_sigma_t * stat, int ntime);

#endif
