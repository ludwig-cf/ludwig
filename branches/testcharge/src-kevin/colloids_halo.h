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
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_HALO_H
#define COLLOIDS_HALO_H

#include "colloids.h"

typedef struct colloid_halo_s colloid_halo_t;

int colloids_halo_create(colloids_info_t * cinfo, colloid_halo_t ** phalo);
void colloids_halo_free(colloid_halo_t * halo);
int colloids_halo_state(colloids_info_t * cinfo);
int colloids_halo_dim(colloid_halo_t * halo, int dim);
int colloids_halo_send_count(colloid_halo_t * halo, int dim);

#endif
 
