/*****************************************************************************
 *
 *  field_psi_init_rt.c
 *
 *  Initialisation of surfactant "psi" order parameter field.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "field_psi_init_rt.h"

int field_init_uniform(field_t * psi, double psi0);

/*****************************************************************************
 *
 *  field_psi_init_rt
 *
 *  Read the initial coice from the input.
 *
 *  Any additional information must be supplied via the structure
 *  field_psi_info_t parameters.
 *
 *****************************************************************************/

int field_psi_init_rt(pe_t * pe, rt_t * rt, field_psi_info_t param,
		      field_t * psi) {

  int p;
  char value[BUFSIZ];

  p = rt_string_parameter(rt, "psi_initialisation", value, BUFSIZ);

  if (p == 0) pe_fatal(pe, "Please specify psi_initialisation in input\n");

  if (strcmp(value, "uniform") == 0) {
    double psi0;

    pe_info(pe, "Initialising psi to a uniform value psi0\n");

    p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
    if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");

    pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
    field_init_uniform(psi, psi0);
    return 0;
  }

  pe_fatal(pe, "Initial psi choice not recognised: %s\n", value);

  return 0;
}

/*****************************************************************************
 *
 *  field_init_uniform
 *
 *****************************************************************************/

int field_init_uniform(field_t * field, double value) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(field);

  cs_nlocal(field->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(field->cs, ic, jc, kc);
	field_scalar_set(field, index, value);

      }
    }
  }

  return 0;
}
