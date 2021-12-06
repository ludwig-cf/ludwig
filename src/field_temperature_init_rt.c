/****************************************************************************
 *
 *  field_temperature_init_rt.c
 *
 *  Run time initialisation for the composition field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <string.h>

#include "io_harness.h"
#include "field_temperature_init_rt.h"

/*****************************************************************************
 *
 *  field_temperature_init_rt
 *
 *  Initial choices for temperature.
 *
 *  Any information required from the free energy is via
 *  the parameters p.
 *
 *****************************************************************************/

int field_temperature_init_rt(pe_t * pe, rt_t * rt, field_temperature_info_t param, field_t * temperature, map_t * map) {

  int p;
  char value[BUFSIZ];
  double Tc;

  assert(pe);
  assert(rt);
  assert(temperature);

  p = rt_string_parameter(rt, "temperature_initialisation", value, BUFSIZ);

  /* Has to be zero everywhere (because initialization of the free energy is done at T = 0... because temperature field is initialized after the free energy) */

  if (p != 0 && strcmp(value, "uniform") == 0) {
    int ihave_T0;
    double T0;

    pe_info(pe, "Initialising temperature as uniform T0\n");

    ihave_T0 = rt_double_parameter(rt, "T0", &T0);
    if (ihave_T0 == 0) pe_fatal(pe, "Please define T0 in the input!\n");

    pe_info(pe, "Initial value T0: %14.7e\n", T0);

    field_temperature_init_uniform(temperature, T0);
  }

  if (p != 0 && strcmp(value, "drop") == 0) {
    double phistar, xi, radius;      /* "Amplitude", can be negative. */

    pe_info(pe, "Initialising temperature as droplet\n");

    rt_double_parameter(rt, "temperature_init_drop_radius", &radius);
    rt_double_parameter(rt, "temperature_init_drop_amplitude", &phistar);
    rt_double_parameter(rt, "temperature_init_drop_xi", &xi);
    pe_info(pe, "Initialising droplet radius:     %14.7e\n", radius);
    pe_info(pe, "Initialising droplet amplitude:  %14.7e\n", phistar);
    pe_info(pe, "Initialising droplet xi:  %14.7e\n", xi);
    field_temperature_init_drop(temperature, xi, radius, phistar);
  }

  /* Set temperature of the colloid (order matter because it overwrites
	temperature at colloid nodes */
  
  rt_double_parameter(rt, "colloid_temperature", &Tc);
  pe_info(pe, "Initialising colloid temperature to %14.7e\n", Tc);
  field_temperature_init_solid(temperature, map, Tc);

  return 0;
}
