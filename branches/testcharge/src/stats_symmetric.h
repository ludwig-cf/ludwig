/*****************************************************************************
 *
 *  stats_symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef STATS_SYMMETRIC_H
#define STATS_SYMMETRIC_H

#include "field_grad.h"
#include "map.h"

int stats_symmetric_length(field_grad_t * dphi, map_t * map, int timestep);

#endif
