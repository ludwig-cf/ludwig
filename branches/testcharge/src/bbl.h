/*****************************************************************************
 *
 *  bbl.h
 *
 *  $Id: bbl.h,v 1.3 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BBL_H
#define BBL_H

#include "colloids.h"

typedef struct bbl_s bbl_t;

int bbl_create(bbl_t ** pobj);
int bbl_free(bbl_t * obj);
int bbl_active_set(bbl_t * bbl, colloids_info_t * cinfo);
int bounce_back_on_links(bbl_t * bbl, colloids_info_t * cinfo);
int bbl_pass0(bbl_t * bbl, colloids_info_t * cinfo);
int bbl_update_colloids(bbl_t * bbl, colloids_info_t * cinfo);

int bbl_surface_stress(bbl_t * bbl, double slocal[3][3]);
int bbl_order_parameter_deficit(bbl_t * bbl, double * delta);

#endif
