/*****************************************************************************
 *
 *  field_grad.h
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

#ifndef FIELD_GRAD_H
#define FIELD_GRAD_H

#include "field.h"
#include "targetDP.h"

typedef struct field_grad_s field_grad_t;

typedef int (* grad_ft)(int nf, const double * field,double * t_field, 
			double * grad, double * t_grad,
                        double * delsq, double * t_delsq,
			char * siteMask,char * t_siteMask
);
typedef int (* dab_ft)(int nf, const double * field, double * dab);

HOST int field_grad_create(field_t * f, int level, field_grad_t ** pobj);
HOST void field_grad_free(field_grad_t * obj);
HOST int field_grad_set(field_grad_t * obj, grad_ft d2, grad_ft d4);
HOST int field_grad_dab_set(field_grad_t * obj, dab_ft dab);
HOST int field_grad_compute(field_grad_t * obj);

HOST int field_grad_scalar_grad(field_grad_t * obj, int index, double grad[3]);
HOST int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq);
HOST int field_grad_scalar_grad_delsq(field_grad_t * obj, int index, double gd[3]);
HOST int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd);
HOST int field_grad_scalar_dab(field_grad_t * obj, int index, double d_ab[3][3]);

__host__ int field_grad_pair_grad(field_grad_t * obj, int index,
				  double grad[2][3]);
__host__ int field_grad_pair_delsq(field_grad_t * obj, int index,
				   double delsq[2]);

HOST int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]);
HOST int field_grad_vector_delsq(field_grad_t * obj, int index, double dp[3]);

HOST int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]);
HOST int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]);

#endif
