/*****************************************************************************
 *
 *  gradient.c
 *
 *  This is an 'abstract' interface for computations of gradients,
 *  usually the order paramter.
 *
 *  An inplementation can provide three things:
 *
 *    gradient_d2       grad_a field and \nabla^2 field
 *    gradient_d4       takes results \nabla^2 and computes
 *                      grad nabla^2 and nabla^4 field
 *    gradient_dyadic   for vector order parameters p_a,
 *                      computes grad_c p_a p_b, and nabla^2 p_a p_b
 *
 *  $Id: gradient.c,v 1.1.2.1 2010-03-30 14:09:27 kevin Exp $
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

#include "gradient.h"

static void (* d2_function_)(const int nop, const double * field,
			     double * grad, double * delsq) = NULL;
static void (* d4_function_)(const int nop, const double * field,
			     double * grad, double * delsq) = NULL;
static void (* dy_function_)(const int nop, const double * field,
			     double * grad, double * delsq) = NULL;

/*****************************************************************************
 *
 *  gradient_d2_set
 *
 *****************************************************************************/

void gradient_d2_set(void (* f)(const int nop, const double * field,
				double * grad, double * delsq)) {
  assert(f);
  d2_function_ = f;
  return;
}

/*****************************************************************************
 *
 *  gradient_d4_set
 *
 *****************************************************************************/

void gradient_d4_set(void (* f)(const int nop, const double * field,
				double * grad, double * delsq)) {
  assert(f);
  d4_function_ = f;
  return;
}

/*****************************************************************************
 *
 *  gradient_d2_dyadic_set
 *
 *****************************************************************************/

void gradient_d2_dyadic_set(void (* f)(const int nop, const double * field,
				       double * grad, double * delsq)) {
  assert(f);
  dy_function_ = f;
  return;
}

/*****************************************************************************
 *
 *  gradient_d2
 *
 *****************************************************************************/

void gradient_d2(const int nop, const double * field,
		 double * grad, double * delsq) {

  assert(d2_function_);
  d2_function_(nop, field, grad, delsq);
  return;
}

/****************************************************************************
 *
 *  gradient_d4
 *
 ****************************************************************************/

void gradient_d4(const int nop, const double * field,
		 double * grad, double * delsq) {

  assert(d4_function_);
  d4_function_(nop, field, grad, delsq);
  return;
}

/****************************************************************************
 *
 *  gradient_d2_dyadic
 *
 ****************************************************************************/

void gradient_d2_dyadic(const int nop, const double * field,
			double * grad, double * delsq) {

  assert(dy_function_);
  dy_function_(nop, field, grad, delsq);
  return;
}
