/*****************************************************************************
 *
 *  interaction.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef INTERACTION_H
#define INTERACTION_H

#include "map.h"
#include "psi.h"
#include "colloids.h"
#include "ewald.h"

typedef enum interact_enum {
  INTERACT_PAIR = 0,
  INTERACT_BOND,
  INTERACT_ANGLE,
  INTERACT_LUBR,
  INTERACT_MAX
} interact_enum_t;

typedef enum interact_stat_enum {
  INTERACT_STAT_VLOCAL = 0,
  INTERACT_STAT_HMINLOCAL,
  INTERACT_STAT_RMINLOCAL,
  INTERACT_STAT_RMAXLOCAL,
  INTERACT_STAT_MAX
} interact_stat_enum_t;

typedef struct interact_s interact_t;

typedef int (*range_ft)(void * self, double * rchmax);
typedef int (*compute_ft)(colloids_info_t * cinfo, void * self);
typedef int (*stat_ft)(void * self, double * stats);

int interact_create(interact_t ** pobj);
void interact_free(interact_t * obj);
int interact_potential_add(interact_t * obj, interact_enum_t it,
			   void * potential, compute_ft compute);
int interact_statistic_add(interact_t * obj, interact_enum_t it,
			   void * potential, stat_ft stats);
int interact_rc_set(interact_t * obj, interact_enum_t it, double rc);
int interact_hc_set(interact_t * obj, interact_enum_t it, double hc);
int interact_range_check(interact_t * obj, colloids_info_t * cinfo);
int interact_compute(interact_t * interact, colloids_info_t * cinfo,
		     map_t * map, psi_t * psi, ewald_t * ewald);
int interact_pairwise(interact_t * interact, colloids_info_t * cinfo);
int interact_bonds(interact_t * obj, colloids_info_t * cinfo);
int interact_angles(interact_t * obj, colloids_info_t * cinfo);
int interact_find_bonds(interact_t * obj, colloids_info_t * cinfo);
int interact_stats(interact_t * obj, colloids_info_t * cinfo);

int colloids_update_forces_zero(colloids_info_t * cinfo);
int colloids_update_forces_external(colloids_info_t * cinfo, psi_t * psi);
int colloids_update_forces_fluid_gravity(colloids_info_t * cinfo, map_t * map);

int colloids_init_ewald_rt(colloids_info_t * cinfo, ewald_t ** pewald);

int colloids_init_halo_range_check(colloids_info_t * cinfo);


#endif
