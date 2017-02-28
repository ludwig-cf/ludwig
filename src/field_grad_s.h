/*****************************************************************************
 *
 *  field_grad_s.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef FIELD_GRAD_S_H
#define FIELD_GRAD_S_H

#include "memory.h"
#include "leesedwards.h"
#include "field_grad.h"

struct field_grad_s {
  pe_t * pe;                /* Parallel environment */
  field_t * field;          /* Reference to the field */
  int nf;                   /* Number of field components */
  int level;                /* Maximum derivative required */
  int nsite;                /* number of sites allocated */
  double * grad;            /* Gradient  \nabla f */
  double * delsq;           /* Laplacian \nabla^2 f */
  double * d_ab;            /* Gradient tensor d_a d_b f */
  double * grad_delsq;      /* Gradient of Laplacian grad \nabla^2 f */
  double * delsq_delsq;     /* Laplacian^2           \nabla^4 f */

  field_grad_t * target;    /* copy of this structure on target */ 

  int (* d2)  (field_grad_t * fgrad);
  int (* d4)  (field_grad_t * fgrad);
  int (* dab) (field_grad_t * fgrad);
};

#endif
