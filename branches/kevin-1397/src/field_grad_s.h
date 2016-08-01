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
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FIELD_GRAD_S_H
#define FIELD_GRAD_S_H

#include "memory.h"
#include "leesedwards.h"
#include "field_grad.h"

struct field_grad_s {
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


/* Commpressed symmetric rank 2 tensor */

#define addr_dab(index, ia) addr_rank1(le_nsites(), NSYMM, index, ia)
#define mem_addr_dab(index, ia) mem_addr_rank1(le_nsites(), NSYMM, index, ia)

#endif
