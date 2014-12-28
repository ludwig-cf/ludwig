/*****************************************************************************
 *
 *  stats_rheology.h
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: stats_rheology.h,v 1.3 2009-10-14 17:16:01 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_RHEOLOGY_H
#define STATS_RHEOLOGY_H

typedef struct stats_rheo_s stats_rheo_t;

#include "coords.h"
#include "model.h"
#include "hydro.h"

int stats_rheology_create(coords_t * cs, stats_rheo_t ** stat);
int stats_rheology_free(stats_rheo_t * stat);

int stats_rheology_stress_profile_accumulate(stats_rheo_t * stat, lb_t * lb,
					     hydro_t * hydro);
int stats_rheology_mean_stress(stats_rheo_t * stat, lb_t * lb,
			       const char * filename);

int stats_rheology_free_energy_density_profile(stats_rheo_t * stat,
					       const char *);
int stats_rheology_stress_profile_zero(stats_rheo_t * stat);
int stats_rheology_stress_profile(stats_rheo_t * stat, const char *);
int stats_rheology_stress_section(stats_rheo_t * stat, const char *);

#endif
