/*****************************************************************************
 *
 *  field_grad.h
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

#ifndef FIELD_GRAD_H
#define FIELD_GRAD_H

#include "pe.h"
#include "field.h"

typedef struct field_grad_s field_grad_t;
typedef int (* grad_ft)(field_grad_t * fgrad);

__host__ int field_grad_create(pe_t * pe, field_t * f, int level,
			       field_grad_t ** pobj);
__host__ void field_grad_free(field_grad_t * obj);
__host__ int field_grad_set(field_grad_t * obj, grad_ft d2, grad_ft d4);
__host__ int field_grad_dab_set(field_grad_t * obj, grad_ft dab);
__host__ int field_grad_compute(field_grad_t * obj);
__host__ int field_grad_memcpy(field_grad_t * obj, int flag);

__host__ __device__ int field_grad_scalar_grad(field_grad_t * obj, int index, double grad[3]);
__host__ __device__ int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq);
__host__ __device__ int field_grad_scalar_grad_delsq(field_grad_t * obj, int index, double gd[3]);
__host__ __device__ int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd);
__host__ __device__ int field_grad_scalar_dab(field_grad_t * obj, int index, double d_ab[3][3]);

__host__ __device__ int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]);
__host__ __device__ int field_grad_vector_delsq(field_grad_t * obj, int index, double dp[3]);

__host__ __device__ int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]);
__host__ __device__ int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]);

#endif
