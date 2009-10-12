/*****************************************************************************
 *
 *  stats_rheology.h
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: stats_rheology.h,v 1.2 2009-10-12 18:17:13 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#ifndef STATS_RHEOLOGY_H
#define STATS_RHEOLOGY_H

void stats_rheology_init(void);
void stats_rheology_finish(void);

void stats_rheology_mean_stress(void);
void stats_rheology_free_energy_density_profile(const char *);
void stats_rheology_stress_profile_zero(void);
void stats_rheology_stress_profile_accumulate(void);
void stats_rheology_stress_profile(const char *);
void stats_rheology_stress_section(const char *);

#endif
