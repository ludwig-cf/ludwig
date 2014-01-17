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

#ifdef OLD_ONLY
void colloid_io_run_time(void);
#else

#include "colloids.h"
#include "colloid_io.h"

int colloid_io_run_time(colloids_info_t * cinfo, colloid_io_t ** pcio);

#endif
#endif
