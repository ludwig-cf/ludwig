/*****************************************************************************
 *
 *  gradient_3d_27pt_solid.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_3D_27PT_SOLID_H
#define GRADIENT_3D_27PT_SOLID_H

#include "map.h"
#include "symmetric.h"
//OFT
#include "symmetric_oft.h"
#include "surfactant_oft.h"
//OFT
#include "field_grad.h"

__host__ int grad_3d_27pt_solid_map_set(map_t * map);
__host__ int grad_3d_27pt_solid_d2(field_grad_t * fg);
//OFT
__host__ int grad_3d_27pt_solid_d2_symm_oft(field_grad_t * fg);
__host__ int grad_3d_27pt_solid_d2_surf_oft(field_grad_t * fg);
__host__ int grad_3d_27pt_solid_symm_oft_set(fe_symm_oft_t * fe);
__host__ int grad_3d_27pt_solid_surf_oft_set(fe_surf_oft_t * fe);
//OFT
__host__ int grad_3d_27pt_solid_fe_set(fe_symm_t * fe);
__host__ int grad_3d_27pt_solid_dab(field_grad_t * fg);

#endif
