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
  double * grad;            /* Gradient              \nabla f (on host)*/
  double * delsq;           /* Laplacian             \nabla^2 f (on target)*/
  double * t_grad;            /* Gradient              \nabla f (on host)*/
  double * t_delsq;           /* Laplacian             \nabla^2 f (on target)*/
  double * d_ab;            /* Gradient tensor d_a d_b f */
  double * grad_delsq;      /* Gradient of Laplacian grad \nabla^2 f */
  double * delsq_delsq;     /* Laplacian^2           \nabla^4 f */

  field_grad_t * tcopy;              /* copy of this structure on target */ 

  int (* d2) (int nf, const double * field, 
	      double * t_field,
	      double * grad,
	      double * t_grad,
	      double * delsq,
	      double * t_delsq
);
  int (* d4) (int nf, const double * field, 
	      double * t_field,
	      double * grad,
	      double * t_grad,
	      double * delsq,
	      double * t_delsq
);
  int (* dab)  (int nf, const double * field, 
	      double * dab);
};
#endif
