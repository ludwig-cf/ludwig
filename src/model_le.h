/*****************************************************************************
 *
 *  model_le.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2024 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MODEL_LE_H
#define LUDWIG_MODEL_LE_H

#include "lb_data.h"
#include "leesedwards.h"

int lb_data_apply_le_boundary_conditions(lb_t * lb, lees_edw_t * le);
int lb_le_init_shear_profile(lb_t * lb, lees_edw_t * le);

#endif
