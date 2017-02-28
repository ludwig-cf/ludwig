/*****************************************************************************
 *
 *  stats_turbulent.h
 *
 *  $Id: stats_turbulent.h,v 1.2 2008-08-24 16:58:10 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_TURBULENT_H
#define LUDWIG_STATS_TURBULENT_H

#include "pe.h"
#include "coords.h"
#include "hydro.h"

typedef struct stats_turb_s stats_turb_t;

int stats_turbulent_create(pe_t * pe, cs_t * cs, stats_turb_t ** pobj);
int stats_turbulent_free(stats_turb_t * stat);
int stats_turbulent_ubar_zero(stats_turb_t * stat);
int stats_turbulent_ubar_accumulate(stats_turb_t * stat, hydro_t * hydro);
int stats_turbulent_ubar_output(stats_turb_t * stat, const char *);

#endif
