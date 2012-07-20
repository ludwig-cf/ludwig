/*****************************************************************************
 *
 *  field_grad_s.h
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

#ifndef FIELD_GRAD_S_H
#define FIELD_GRAD_S_H

#include "field_grad.h"

struct field_grad_s {
  field_t * field;          /* Reference to the field */
  int nf;                   /* Number of field components */
  int level;                /* Maximum derivative required */
  double * grad;            /* Gradient              \nabla f */
  double * delsq;           /* Laplacian             \nabla^2 f */
  double * grad_delsq;      /* Gradient of Laplacian grad \nabla^2 f */
  double * delsq_delsq;     /* Laplacian^2           \nabla^4 f */

  int (* d2) (int nf, const double * field, double * grad, double * delsq);
  int (* d4) (int nf, const double * field, double * grad, double * delsq);
};

#endif
