/*****************************************************************************
 *
 *  field_grad.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FIELD_GRAD_H
#define FIELD_GRAD_H

#include "field.h"
#include "targetDP.h"

typedef struct field_grad_s field_grad_t;
typedef int (* grad_ft)(field_grad_t * fgrad);

__targetHost__ int field_grad_create(field_t * f, int level, field_grad_t ** pobj);
__targetHost__ void field_grad_free(field_grad_t * obj);
__targetHost__ int field_grad_set(field_grad_t * obj, grad_ft d2, grad_ft d4);
__targetHost__ int field_grad_dab_set(field_grad_t * obj, grad_ft dab);
__targetHost__ int field_grad_compute(field_grad_t * obj);

__targetHost__ int field_grad_scalar_grad(field_grad_t * obj, int index, double grad[3]);
__targetHost__ int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq);
__targetHost__ int field_grad_scalar_grad_delsq(field_grad_t * obj, int index, double gd[3]);
__targetHost__ int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd);
__targetHost__ int field_grad_scalar_dab(field_grad_t * obj, int index, double d_ab[3][3]);

__targetHost__ int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]);
__targetHost__ int field_grad_vector_delsq(field_grad_t * obj, int index, double dp[3]);

__targetHost__ int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]);
__targetHost__  int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]);

#endif
