/*****************************************************************************
 *
 *  interaction.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2020 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_INTERACTION_H
#define LUDWIG_INTERACTION_H

#include "map.h"
#include "psi.h"
#include "colloids.h"
#include "ewald.h"

#include "field.h"
#include "field_s.h"

//CHANGE1
typedef enum interact_enum {
  INTERACT_PAIR = 0,
  INTERACT_BOND,
  INTERACT_BOND_HARMONIC,
  INTERACT_BOND_HARMONIC2,
  INTERACT_BOND_HARMONIC3,
  INTERACT_MESH_HARMONIC,
  INTERACT_ANGLE,
  INTERACT_ANGLE_HARMONIC,
  INTERACT_ANGLE_DIHEDRAL,
  INTERACT_LUBR,
  INTERACT_WALL,
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

int interact_create(pe_t * pe, cs_t * cs, interact_t ** pobj);
void interact_free(interact_t * obj);
int interact_potential_add(interact_t * obj, interact_enum_t it,
			   void * potential, compute_ft compute);
int interact_statistic_add(interact_t * obj, interact_enum_t it,
			   void * potential, stat_ft stats);
int interact_rc_set(interact_t * obj, interact_enum_t it, double rc);
int interact_hc_set(interact_t * obj, interact_enum_t it, double hc);
int interact_range_check(interact_t * obj, colloids_info_t * cinfo);

/* -----> CHEMOVESICLE V2 */
/* Now requires rt to know whether phi_subgrid_switch is on */
int interact_compute(interact_t * interact, colloids_info_t * cinfo,
		     map_t * map, psi_t * psi, ewald_t * ewald,
			field_t * phi,
			field_t * subgrid_potential, rt_t * rt, field_t * flux_mask);
/* <----- */

int interact_pairwise(interact_t * interact, colloids_info_t * cinfo);
int interact_wall(interact_t * interact, colloids_info_t * cinfo);
int interact_bonds(interact_t * obj, colloids_info_t * cinfo);
int interact_bonds_harmonic(interact_t * obj, colloids_info_t * cinfo);
int interact_bonds_harmonic2(interact_t * obj, colloids_info_t * cinfo);
int interact_bonds_harmonic3(interact_t * obj, colloids_info_t * cinfo);
int interact_mesh_harmonic(interact_t * obj, colloids_info_t * cinfo);

int interact_angles(interact_t * obj, colloids_info_t * cinfo);
int interact_angles_harmonic(interact_t * obj, colloids_info_t * cinfo);
int interact_angles_dihedral(interact_t * obj, colloids_info_t * cinfo);

int colloids_update_discrete_forces_phi(colloids_info_t * cinfo, field_t * phi, field_t * subgrid_potential, field_t * flux_mask, rt_t * t);
int colloids_update_discrete_forces_phi_old(colloids_info_t * cinfo, field_t * phi, field_t * subgrid_potential);
int colloids_update_analytic_forces_phi(colloids_info_t * cinfo, field_t * phi, field_t * subgrid_potential);

int interact_find_bonds(interact_t * obj, colloids_info_t * cinfo);
int interact_find_bonds2(interact_t * obj, colloids_info_t * cinfo);
int interact_find_bonds3(interact_t * obj, colloids_info_t * cinfo);
int interact_find_bonds_mesh(interact_t * obj, colloids_info_t * cinfo);

int interact_find_bonds_all(interact_t * obj, colloids_info_t * cinfo, int nx);
int interact_find_bonds_all2(interact_t * obj, colloids_info_t * cinfo, int nx);
int interact_find_bonds_all3(interact_t * obj, colloids_info_t * cinfo, int nx);
int interact_find_bonds_all_mesh(interact_t * obj, colloids_info_t * cinfo, int nx);

int interact_stats(interact_t * obj, colloids_info_t * cinfo);
int interact_hcmax(interact_t * obj, double * hcmax);
int interact_rcmax(interact_t * obj, double * rcmax);

int colloids_update_forces_zero(colloids_info_t * cinfo);
int colloids_update_forces_external(colloids_info_t * cinfo, psi_t * psi);
int colloids_update_forces_fluid_gravity(colloids_info_t * cinfo, map_t * map);
int colloids_update_forces_fluid_driven(colloids_info_t * cinfo, map_t * map);
int colloids_update_forces_ext(colloids_info_t * cinfo);

#endif
