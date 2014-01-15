/*****************************************************************************
 *
 *  propagation_ode.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PROPAGATION_ODE_H
#define PROPAGATION_ODE_H

#include "hydro.h"

void   propagation_ode(hydro_t * hydro);
void   propagation_ode_init(void);
double propagation_ode_get_tstep(void);
enum   propagation_ode_integrator_type {RK2, RK4};

#endif

