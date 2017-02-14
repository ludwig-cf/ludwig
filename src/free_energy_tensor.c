/*****************************************************************************
 *
 *  free_energy_tensor.c
 *
 *  Abstract 'class' for free energies having tensor order parameter.
 *  A concrete implementation must provide at least an implementation
 *  of the function
 *
 *  void fe_t_molecular_field(const int index, double h[3][3])
 *
 *  to return the (vector) molecular field h.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  (c) 2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include "free_energy_tensor.h"

static void (* fp_h_)(const int index, double h[3][3]) = NULL;

/*****************************************************************************
 *
 *  fe_t_molecular_field_set
 *
 *  Set the function pointer for the molecular field.
 *
 *****************************************************************************/

void fe_t_molecular_field_set(void (* f)(const int index, double h[3][3])) {

  assert(f);
  fp_h_ = f;
  return;
}

/*****************************************************************************
 *
 *  fe_t_moleuclar_field(void)
 *
 *  Return the pointer to the molecular field function.
 *
 *****************************************************************************/

void (* fe_t_molecular_field(void))(const int index, double h[3][3]) {

  assert(fp_h_);
  return fp_h_;
}
