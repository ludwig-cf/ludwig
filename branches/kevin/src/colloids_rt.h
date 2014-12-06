/*****************************************************************************
 *
 *  colloids_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef COLLOIDS_RT_H
#define COLLOIDS_RT_H

#include "runtime.h"
#include "colloids.h"
#include "colloid_io.h"
#include "interaction.h"
#include "map.h"
#include "ewald.h"

int colloids_init_rt(rt_t * rt, colloids_info_t ** pinfo, colloid_io_t ** cio,
		     interact_t ** interact, map_t * map);
int colloids_init_ewald_rt(rt_t * rt, colloids_info_t * cinfo,
			   ewald_t ** pewald);
int colloids_init_halo_range_check(colloids_info_t * cinfo);

#endif
