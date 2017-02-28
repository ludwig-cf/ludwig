/*****************************************************************************
 *
 *  stats_rheology.h
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: stats_rheology.h,v 1.3 2009-10-14 17:16:01 kevin Exp $
 *
 *  (c) 2009-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STATS_RHEOLOGY_H
#define LUDWIG_STATS_RHEOLOGY_H

#include "free_energy.h"
#include "model.h"
#include "hydro.h"

typedef struct stats_rheo_s stats_rheo_t;

int stats_rheology_create(pe_t * pe, cs_t * cs, stats_rheo_t ** prheo);
int stats_rheology_free(stats_rheo_t * rheo);

int stats_rheology_stress_profile_accumulate(stats_rheo_t * rheo,
					     lb_t * lb, fe_t * fe,
					     hydro_t * hydro);
int stats_rheology_mean_stress(lb_t * lb, fe_t * fe,
			       const char * filename);

int stats_rheology_free_energy_density_profile(stats_rheo_t * rheo, fe_t * fe,
					       const char *);
int stats_rheology_stress_profile_zero(stats_rheo_t * rheo);
int stats_rheology_stress_profile(stats_rheo_t * rheo, const char *);
int stats_rheology_stress_section(stats_rheo_t * rheo, const char *);

#endif
