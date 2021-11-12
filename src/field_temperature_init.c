/****************************************************************************
 *
 *  field_temperature_init.c
 *
 *  Initial compositional order parameter configurations.
 *  Independent of the free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "noise.h"
#include "util.h"
#include "field_s.h"
#include "field_temperature_init.h"

/*****************************************************************************
 *
 *  field_temperature_init_uniform
 *
 *  Uniform field = T0; T0 should be consistent with phys_t object.
 *
 *****************************************************************************/

int field_temperature_init_uniform(field_t * temperature, double T0) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(temperature);

  cs_nlocal(temperature->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(temperature->cs, ic, jc, kc);

	field_scalar_set(temperature, index, T0);
      }
    }
  }

  return 0;
}

