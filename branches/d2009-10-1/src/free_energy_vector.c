/*****************************************************************************
 *
 *  free_energy_vector.c
 *
 *  Abstract 'class' for free energies having vector order parameter.
 *  A concrete implementation must provide at least an implementation
 *  of the function
 *
 *  void fe_v_molecular_field(const int index, double h[3])
 *
 *  to return the (vector) molecular field h.
 *
 *  Here we also store
 *
 *     zeta      activity parameter, e.g., for polar active gels
 *     lambda    material parameter (+ve for rod-like molecules)
 *               |lambda| < 1 gives flow-aligning
 *               |lambda| > 1 gives flow tumbling
 *
 *  These quantities are zero by default.
 *
 *  $Id: free_energy_vector.c,v 1.1.2.2 2010-03-26 05:44:44 kevin Exp $
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
#include "free_energy_vector.h"

static double lambda_ = 0.0;

static void (* fp_h_)(const int index, double h[3]) = NULL;

/*****************************************************************************
 *
 *  fe_v_molecular_field_set
 *
 *  Set the function pointer for the molecular field.
 *
 *****************************************************************************/

void fe_v_molecular_field_set(void (* f)(const int index, double h[3])) {

  assert(f);
  fp_h_ = f;
  return;
}

/*****************************************************************************
 *
 *  fe_v_moleuclar_field(void)
 *
 *  Return the pointer to the molecular field function.
 *
 *****************************************************************************/

void (* fe_v_molecular_field(void))(const int index, double h[3]) {

  assert(fp_h_);
  return fp_h_;
}

/*****************************************************************************
 *
 *  fe_v_lambda_set
 *
 *****************************************************************************/

void fe_v_lamda_set(const double lambda_new) {

  lambda_ = lambda_new;
  return;
}

/*****************************************************************************
 *
 *  fe_v_lambda
 *
 *****************************************************************************/

double fe_v_lambda(void) {

  return lambda_;
}
