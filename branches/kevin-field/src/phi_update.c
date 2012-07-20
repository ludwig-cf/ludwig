/*****************************************************************************
 *
 *  phi_update.c
 *
 *  'Abstract' class for update of the order parameter.
 *
 *  The actual function which makes the update (i.e., the dynamics)
 *  must be specified by a call 
 *  $Id: phi_update.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "hydro.h"
#include "phi_update.h"

static int (* fp_phi_dynamics_function_)(hydro_t * hydro) = NULL;

/*****************************************************************************
 *
 *  phi_update_dynamics
 *
 *****************************************************************************/

int phi_update_dynamics(hydro_t * hydro) {

  if (fp_phi_dynamics_function_) {

    fp_phi_dynamics_function_(hydro);

  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_update_set
 *
 *****************************************************************************/

int phi_update_set(phi_dynamics_update_ft f) {

  assert(f);
  fp_phi_dynamics_function_ = f;

  return 0;
}
