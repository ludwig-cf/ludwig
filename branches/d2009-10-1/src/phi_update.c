/*****************************************************************************
 *
 *  phi_update.c
 *
 *  'Abstract' class for update of the order parameter.
 *
 *  The actual function which makes the update (i.e., the dynamics)
 *  must be specified by a call 
 *  $Id: phi_update.c,v 1.1.2.1 2010-03-26 08:41:50 kevin Exp $
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
#include "timer.h"
#include "phi_update.h"

static void (* fp_phi_dynamics_function_)(void) = NULL;

/*****************************************************************************
 *
 *  phi_update_dynamics
 *
 *****************************************************************************/

void phi_update_dynamics(void) {

  if (fp_phi_dynamics_function_) {

    TIMER_start(TIMER_ORDER_PARAMETER_UPDATE);
    fp_phi_dynamics_function_();
    TIMER_stop(TIMER_ORDER_PARAMETER_UPDATE);

  }

  return;
}

/*****************************************************************************
 *
 *  phi_update_set
 *
 *****************************************************************************/

void phi_update_set(void (* f)(void)) {

  assert(f);
  fp_phi_dynamics_function_ = f;

  return;
}
