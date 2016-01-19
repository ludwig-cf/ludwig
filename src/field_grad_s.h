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


#ifndef OLD_SHIT

#include "memory.h"

/* Commpressed symmetric rank 2 tensor */

#define addr_dab(nsites, index, ia) addr_rank1(nsites, NSYMM, index, ia)
#define vaddr_dab(nsites, index, ia, iv) vaddr_rank1(nsites, NSYMM, index, ia, iv)

#else

/* array of structures */
#define ADDR_FGRD(nsite, nfield, index, ifield, idir)	\
  ((nfield)*3*(index) + (ifield)*3+(idir))


/* structure of arrays */
#define ADDR_FGRD_R(nsite, nfield, index, ifield,idir)	\
  ((nsite)*3*(ifield) + (nsite)*(idir) + (index))

#ifdef LB_DATA_SOA
#define FGRDADR ADDR_FGRD_R
#else
#define FGRDADR ADDR_FGRD

#endif
#endif
#endif
