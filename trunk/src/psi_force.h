/*****************************************************************************
 *
 *  psi_force.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013 The University of Edinburgh
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    
 *****************************************************************************/

#ifndef PSI_FORCE_H
#define PSI_FORCE_H

#include "psi.h"
#include "hydro.h"
#include "colloids.h"
#include "map.h"

int psi_force_gradmu(psi_t * psi,  hydro_t * hydro,
			map_t * map, colloids_info_t * cinfo);
int psi_force_divstress(psi_t * psi, hydro_t * hydro, 
			colloids_info_t * cinfo);
int psi_force_divstress_d3qx(psi_t * psi, hydro_t * hydro, 
			map_t * map, colloids_info_t * cinfo);
int psi_force_divstress_one_sided_d3qx(psi_t * psi, hydro_t * hydro, 
			map_t * map, colloids_info_t * cinfo);
int psi_force_divergence_set(const int flag);
int psi_force_is_divergence(int * flag);
#endif
