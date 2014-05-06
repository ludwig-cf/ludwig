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

int bounce_back_on_links(colloids_info_t * cinfo);
int bbl_pass0(colloids_info_t * cinfo);
int bbl_update_colloids(colloids_info_t * cinfo);

void bbl_surface_stress(void);
void bbl_active_on_set(void);
int  bbl_active_on(void);
int bbl_order_parameter_deficit(double * delta);

#endif
