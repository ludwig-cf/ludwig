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

#include "hydro.h"
#include "field.h"
#include "map.h"
#include "psi.h"
#include "colloids.h"

int COLL_update(hydro_t * hydro, map_t * map, field_t * phi, field_t * p,
		field_t * q, psi_t * psi);
#ifdef OLD_ONLY
void COLL_init(map_t * map);
double    colloid_forces_ahmax(void);
#else

#include "colloid_io.h"
#include "ewald.h"

int colloids_init_rt(colloids_info_t ** pinfo, colloid_io_t ** cio,
		     map_t * map);
int colloids_update_position(colloids_info_t * cinfo);
int colloids_update_forces(colloids_info_t * cinfo, map_t * map, psi_t * psi,
			   ewald_t * ewald);
int colloids_update_forces_zero(colloids_info_t * cinfo);
int colloids_update_forces_external(colloids_info_t * cinfo, psi_t * psi);
int colloids_update_forces_fluid_gravity(colloids_info_t * cinfo, map_t * map);
int colloids_forces_ahmax(colloids_info_t * cinfo, double * ahmax);

int colloids_init_ewald_rt(colloids_info_t * cinfo, ewald_t ** pewald);
#endif

#endif
