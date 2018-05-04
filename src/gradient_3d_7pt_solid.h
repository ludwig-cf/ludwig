/*****************************************************************************
 *
 *  gradient_3d_7pt_solid.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2017 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_3D_7PT_SOLID_H
#define GRADIENT_3D_7PT_SOLID_H

#include "pe.h"
#include "coords.h"
#include "map.h"
#include "colloids.h"
#include "field_grad.h"
#include "lc_droplet.h"

typedef struct grad_lc_anch_s grad_lc_anch_t;

__host__ int grad_lc_anch_create(pe_t * pe, cs_t * cs, map_t * map,
				 field_t * phi, colloids_info_t * cinfo,
				 fe_lc_t * fe, grad_lc_anch_t ** p);
__host__ int grad_3d_7pt_solid_d2(field_grad_t * fg);
__host__ int grad_3d_7pt_solid_dab(field_grad_t * fg);
__host__ int grad_3d_7pt_solid_set(map_t * map, colloids_info_t * cinfo);

#endif
