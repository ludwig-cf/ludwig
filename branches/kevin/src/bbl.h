/*****************************************************************************
 *
 *  bbl.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BBL_H
#define BBL_H

#include "coords.h"
#include "colloids.h"
#include "model.h"
#include "wall.h"

typedef struct bbl_s bbl_t;

int bbl_create(coords_t * cs, lb_t * lb, bbl_t ** pobj);
int bbl_free(bbl_t * obj);

int bounce_back_on_links(bbl_t * bbl, lb_t * lb, wall_t * wall,
			 colloids_info_t * cinfo);
int bbl_pass0(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo);

int bbl_active_set(bbl_t * bbl, colloids_info_t * cinfo);
int bbl_update_colloids(bbl_t * bbl, wall_t * wall, colloids_info_t * cinfo);

int bbl_surface_stress(bbl_t * bbl, double slocal[3][3]);
int bbl_order_parameter_deficit(bbl_t * bbl, double * delta);

#endif
