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

int COLL_init(map_t * map);
int COLL_update(hydro_t * hydro, map_t * map, field_t * phi, field_t * p,
		field_t * q, psi_t * psi);
void      colloid_gravity(double f[3]);
void      colloid_gravity_set(const double f[3]);
double    colloid_forces_ahmax(void);

#endif
