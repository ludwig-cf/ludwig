/*****************************************************************************
 *
 *  phi_update.h
 *
 *  $Id: phi_update.h,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_UPDATE_H
#define PHI_UPDATE_H

#include "hydro.h"

typedef int (*phi_dynamics_update_ft)(hydro_t * hydro);

int phi_update_dynamics(hydro_t * hydro);
int phi_update_set(phi_dynamics_update_ft);

#endif
