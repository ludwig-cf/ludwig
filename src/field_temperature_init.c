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
 *  field_temperature_init_solid
 *
 *  Initialise the temperature of colloid
 *
 *****************************************************************************/

int field_temperature_init_solid(field_t * temperature, map_t * map, double Tc) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(temperature);
  assert(map);

  cs_nlocal(temperature->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(temperature->cs, ic, jc, kc);
	if (map->status[index] == MAP_COLLOID) {
	  temperature->data[addr_rank0(temperature->nsites, index)] = Tc;
	}
      }
    }
  }
  return 0;
}


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



/*****************************************************************************
 *
 *  field_temperature_init_drop
 *
 *  Droplet based on a profile temperature(r) = temperaturestar tanh (r-r0)/xi
 *  with r0 the centre of the system.
 *
 *****************************************************************************/

int field_temperature_init_drop(field_t * temperature, double xi, double radius, double phistar) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double ltot[3];
  double position[3];
  double centre[3];
  double phival, r, rxi0;

  assert(temperature);

  cs_nlocal(temperature->cs, nlocal);
  cs_nlocal_offset(temperature->cs, noffset);
  cs_ltot(temperature->cs, ltot);

  rxi0 = 1.0/xi;

  centre[X] = 0.5*ltot[X];
  centre[Y] = 0.5*ltot[Y];
  centre[Z] = 0.5*ltot[Z];

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(temperature->cs, ic, jc, kc);
        position[X] = 1.0*(noffset[X] + ic) - centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - centre[Z];

        r = sqrt(dot_product(position, position));

        phival = phistar*tanh(rxi0*(r - radius));
        field_scalar_set(temperature, index, phival);
      }
    }
  }

  return 0;
}

