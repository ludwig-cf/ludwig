/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter field to the 'g' distribution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "phi_lb_coupler.h"

/*****************************************************************************
 *
 *  phi_lb_to_field
 *
 *****************************************************************************/

int phi_lb_to_field(field_t * phi, lb_t  *lb) {

  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(lb);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	lb_0th_moment(lb, index, 1, &phi0);
	field_scalar_set(phi, index, phi0);

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_lb_from_field
 *
 *  Move the scalar order parameter into the non-propagating part
 *  of the distribution, and set other elements of distribution to
 *  zero.
 *
 *****************************************************************************/

int phi_lb_from_field(field_t * phi, lb_t * lb) {

  int p;
  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(lb);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_scalar(phi, index, &phi0);

	lb_f_set(lb, index, 0, 1, phi0);
	for (p = 1; p < NVEL; p++) {
	  lb_f_set(lb, index, p, 1, 0.0);
	}

      }
    }
  }

  return 0;
}
