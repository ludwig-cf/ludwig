/*****************************************************************************
 *
 *  colloid_io_rt.h
 *
 *  $Id: colloid_io_rt.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOID_IO_RT_H
#define COLLOID_IO_RT_H

#include "coords.h"
#include "runtime.h"
#include "colloids.h"
#include "colloid_io.h"

int colloid_io_run_time(rt_t * rt, coords_t * cs, colloids_info_t * cinfo,
			colloid_io_t ** pcio);

#endif
