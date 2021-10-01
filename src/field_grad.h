/*****************************************************************************
 *
 *  field_grad.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_GRAD_H
#define LUDWIG_FIELD_GRAD_H

#include "pe.h"
#include "field.h"

typedef struct field_grad_s field_grad_t;
typedef int (* grad_ft)(field_grad_t * fgrad);

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


__host__ int field_grad_create(pe_t * pe, field_t * f, int level,
			       field_grad_t ** pobj);
__host__ void field_grad_free(field_grad_t * obj);
__host__ int field_grad_set(field_grad_t * obj, grad_ft d2, grad_ft d4);
__host__ int field_grad_dab_set(field_grad_t * obj, grad_ft dab);
__host__ int field_grad_compute(field_grad_t * obj);
__host__ int field_grad_memcpy(field_grad_t * obj, tdpMemcpyKind flag);

__host__ __device__ int field_grad_scalar_grad(field_grad_t * obj, int index, double grad[3]);
__host__ __device__ int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq);
__host__ __device__ int field_grad_scalar_grad_delsq(field_grad_t * obj, int index, double gd[3]);
__host__ __device__ int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd);
__host__ __device__ int field_grad_scalar_dab(field_grad_t * obj, int index, double d_ab[3][3]);

__host__ __device__ int field_grad_pair_grad(field_grad_t * obj, int index, double grad[2][3]);
__host__ __device__ int field_grad_pair_delsq(field_grad_t * obj, int index, double * delsq);

__host__ __device__ int field_grad_pair_grad_set(field_grad_t * obj, int index, const double grad[2][3]);
__host__ __device__ int field_grad_pair_delsq_set(field_grad_t * obj, int index, const double * delsq);

__host__ __device__ int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]);
__host__ __device__ int field_grad_vector_delsq(field_grad_t * obj, int index, double dp[3]);

__host__ __device__ int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]);
__host__ __device__ int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]);

#endif
