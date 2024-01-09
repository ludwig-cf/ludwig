/*****************************************************************************
 *
 *  phi_stats.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_STATS_H
#define LUDWIG_PHI_STATS_H

#include <mpi.h>
#include "field.h"
#include "map.h"
#include "bbl.h"

int stats_field_info(field_t * obj, map_t * map);
int stats_field_info_bbl(field_t * obj, map_t * map, bbl_t * bbl);

#endif
