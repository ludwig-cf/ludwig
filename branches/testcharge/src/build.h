/*****************************************************************************
 *
 *  build.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BUILD_H
#define BUILD_H

#include "field.h"
#include "map.h"
#include "psi.h"

int build_update_links(map_t * map);
int build_update_map(map_t * map);
void COLL_init_coordinates(void);
int build_remove_or_replace_fluid(field_t * fphi, field_t * fp, field_t * fq,
				  psi_t * psi);

int build_conservation(field_t * phi, psi_t * psi);

#endif
