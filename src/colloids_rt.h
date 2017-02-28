/*****************************************************************************
 *
 *  colloids_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOIDS_RT_H
#define LUDWIG_COLLOIDS_RT_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "colloids.h"
#include "colloid_io.h"
#include "interaction.h"
#include "map.h"
#include "ewald.h"
#include "wall.h"

int colloids_init_rt(pe_t * pe, rt_t * rt, cs_t * cs, colloids_info_t ** pinfo,
		     colloid_io_t ** cio,
		     interact_t ** interact, wall_t * wall, map_t * map);
int colloids_init_ewald_rt(pe_t * pe, rt_t * rt, cs_t * cs,
			   colloids_info_t * cinfo, ewald_t ** pewald);
int colloids_init_halo_range_check(pe_t * pe, cs_t * cs,
				   colloids_info_t * cinfo);

#endif
