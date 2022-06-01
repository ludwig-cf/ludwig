/*****************************************************************************
 *
 *  gradient_s7_anchoring.h
 *
 *  Gradient with liquid crystal anchoring boundary conditions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2022 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_S7_ANCHORING_H
#define GRADIENT_S7_ANCHORING_H

#include "pe.h"
#include "coords.h"
#include "map.h"
#include "field_grad.h"
#include "blue_phase.h"

typedef struct grad_s7_anch_s grad_s7_anch_t;

__host__ int grad_s7_anchoring_create(pe_t * pe, cs_t * cs, map_t * map,
				      fe_lc_t * fe, grad_s7_anch_t ** p);
__host__ int grad_s7_anchoring_map_set(map_t * map);
__host__ int grad_s7_anchoring_d2(field_grad_t * fg);

#endif
