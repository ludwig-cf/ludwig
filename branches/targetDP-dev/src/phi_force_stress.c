/*****************************************************************************
 *
 *  phi_force_stress.c
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "phi_force_stress.h"

static double * pth_;

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

void phi_force_stress_compute(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nextra = 1;

  double pth_local[3][3];
  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	chemical_stress(index, pth_local);
	phi_force_stress_set(index, pth_local);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

void phi_force_stress_set(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth_[9*index + n++] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

void phi_force_stress(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth_[9*index + n++];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_allocate
 *
 *****************************************************************************/

void phi_force_stress_allocate() {

  int n;

  assert(coords_nhalo() >= 2);

  n = coords_nsites();

  pth_ = (double *) malloc(9*n*sizeof(double));
  if (pth_ == NULL) fatal("malloc(pth_) failed\n");

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_free
 *
 *****************************************************************************/

void phi_force_stress_free() {

  free(pth_);

  return;
}
