/*****************************************************************************
 *
 *  free_energy
 *
 *  This is an 'abstract' free energy interface.
 *
 *  Any concrete implementation should implement one or more of the
 *  following functions:
 *
 *  fe_fed_ft      computes free energy density
 *  fe_mu_ft       computes one or more chemical potentials
 *  fe_str_ft      computes thermodynamic stress
 *  fe_mu_solv_ft  computes one or more solvation chemical potentials
 *  fe_hvector_ft  computes molecular field (vector order parameters)
 *  fe_htensor_ft  computes molecular field (tensor order parameters)
 *
 *  The choice of free energy is also related to the form of the order
 *  parameter, and sets the highest order of spatial derivatives of
 *  the order parameter required for the free energy calculations.
 *
 *  A free energy must implement at least a free energy density, a stress,
 *  and an approproiate chemical potential or molecular field.
 *  A destructor (free) should also be defined.
 *
 *  C casts can take place between the abstract and a concrete
 *  pointers, e.g.,
 *
 *  fe_t * fe;
 *  fe_foo_t * fe_derived;
 *
 *  fe = (fe_t *) fe_derived;         // upcast
 *  fe_derived = (fe_derived *) fe;   // downcast
 *
 *  No objects of the abstract type fe_t should be instantiated (there
 *  is no constructor in this class). A virtual destructor is supplied.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "fe_s.h"

/*****************************************************************************
 *
 *  fe_free
 *
 *****************************************************************************/

int fe_free(fe_t * fe) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->free);

  fe->vtable->free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_fed
 *
 *****************************************************************************/

int fe_fed(fe_t * fe, int index, double * fed) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->fed);

  return fe->vtable->fed(fe, index, fed);
}

/*****************************************************************************
 *
 *  fe_mu
 *
 *****************************************************************************/

int fe_mu(fe_t * fe, int index, double * mu) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->mu);

  return fe->vtable->mu(fe, index, mu);
}

/*****************************************************************************
 *
 *  fe_str
 *
 *****************************************************************************/

int fe_str(fe_t * fe, int index, double s[3][3]) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->stress);

  return fe->vtable->stress(fe, index, s);
}

/*****************************************************************************
 *
 *  fe_hvector
 *
 *****************************************************************************/

int fe_hvector(fe_t * fe, int index, double h[3]) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->hvector);

  return fe->vtable->hvector(fe, index, h);
}

/*****************************************************************************
 *
 *  fe_htensor
 *
 *****************************************************************************/

int fe_htensor(fe_t * fe, int index, double h[3][3]) {

  assert(fe);
  assert(fe->vtable);
  assert(fe->vtable->htensor);

  return fe->vtable->htensor(fe, index, h);
}
