/*****************************************************************************
 *
 *  model_le.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2014 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef MODEL_LE_H
#define MODEL_LE_H

#include "model.h"
#include "leesedwards.h"

int lb_le_apply_boundary_conditions(lb_t * lb, lees_edw_t * le);
int lb_le_init_shear_profile(lb_t * lb, lees_edw_t * le);

#endif
