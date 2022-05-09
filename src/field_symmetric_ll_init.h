/*****************************************************************************
 *
 *  field_symmetric_ll_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_SYMMETRIC_LL_INIT_H
#define LUDWIG_FIELD_SYMMETRIC_LL_INIT_H

#include "field.h"

int field_symmetric_ll_phi_uniform(field_t * phi, double phi0);
int field_symmetric_ll_psi_uniform(field_t * phi, double psi0);

#endif
