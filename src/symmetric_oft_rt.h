/****************************************************************************
 *
 *  symmetric_oft_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_OFT_RT_H
#define SYMMETRIC_OFT_RT_H

#include "pe.h"
#include "runtime.h"
#include "symmetric_oft.h"
#include "map.h" // field sum at the beginning

int fe_symmetric_oft_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe);
int fe_symmetric_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe, 
field_t * phi);
int fe_symmetric_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe, field_t * temperature);

//conservation phi correction
int field_sum_phi_init_rt(field_t * field, map_t * map);

#endif
