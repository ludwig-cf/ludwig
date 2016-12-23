/****************************************************************************
 *
 *  symmetric_rt.h
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

#ifndef SYMMETRIC_RT_H
#define SYMMETRIC_RT_H

#include "pe.h"
#include "runtime.h"
#include "symmetric.h"

int fe_symmetric_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe);
int fe_symmetric_phi_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe,
			     field_t * phi);

#endif
