/*****************************************************************************
 *
 *  gradients_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2019 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_GRADIENT_RT_H
#define LUDWIG_GRADIENT_RT_H

#include "pe.h"
#include "runtime.h"
#include "field_grad.h"
#include "map.h"
#include "colloids.h"

#include "gradient_2d_5pt_fluid.h"
#include "gradient_2d_tomita_fluid.h"
#include "gradient_3d_7pt_fluid.h"
#include "gradient_3d_7pt_solid.h"
#include "gradient_3d_27pt_fluid.h"
#include "gradient_3d_27pt_solid.h"
#include "gradient_3d_ternary_solid.h"

__host__ int gradient_rt_init(pe_t * pe, rt_t * rt, const char * fieldname,
			      field_grad_t * grad, map_t * map,
			      colloids_info_t * cinfo);

#endif
