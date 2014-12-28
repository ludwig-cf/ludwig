/*****************************************************************************
 *
 *  colloids_halo.h
 *
 *  $Id: colloids_halo.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_HALO_H
#define COLLOIDS_HALO_H

#include "coords.h"
#include "colloids.h"

typedef struct colloid_halo_s colloid_halo_t;

int colloids_halo_create(coords_t * cs, colloids_info_t * cinfo,
			 colloid_halo_t ** phalo);
int colloids_halo_free(colloid_halo_t * halo);
int colloids_halo_state(coords_t * cs, colloids_info_t * cinfo);
int colloids_halo_dim(colloid_halo_t * halo, int dim);
int colloids_halo_send_count(colloid_halo_t * halo, int dim, int * nreturn);

#endif
 
