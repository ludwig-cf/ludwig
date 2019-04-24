/*****************************************************************************
 *
 *  field_grad.c
 *
 *  Gradients (not just "grad" in the mathematical sense) of a field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_s.h"
#include "field_grad_s.h"

static int field_grad_init(field_grad_t * obj);

/*****************************************************************************
 *
 *  field_grad_create
 *
 *****************************************************************************/

__host__ int field_grad_create(pe_t * pe, field_t * f, int level,
			       field_grad_t ** pobj) {

  field_grad_t * obj =  (field_grad_t *) NULL;

  assert(pe);
  assert(f);
  assert(pobj);

  obj = (field_grad_t*) calloc(1, sizeof(field_grad_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(field_grad_t) failed\n");

  obj->pe = pe;
  obj->field = f;
  obj->level = level;

  field_nf(f, &obj->nf);
  assert(obj->nf > 0);

  field_grad_init(obj);
  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_init
 *
 *****************************************************************************/

static int field_grad_init(field_grad_t * obj) {

  int ndevice;
  int nsites;
  double * tmp;

  assert(obj);

  nsites = obj->field->nsites;
  obj->nsite = nsites;

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    tdpMalloc((void **) &obj->target, sizeof(field_grad_t));
    tdpMemset(obj->target, 0, sizeof(field_grad_t));
    tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int), tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->field, &obj->field->target, sizeof(field_t *),
	      tdpMemcpyHostToDevice);
  }

  if (obj->level >= 2) {

    obj->grad = (double *) calloc(NVECTOR*obj->nf*nsites, sizeof(double));
    obj->delsq = (double *) calloc(obj->nf*nsites, sizeof(double));

    if (obj->grad == NULL) pe_fatal(obj->pe, "calloc(field_grad->grad) failed");
    if (obj->delsq == NULL) pe_fatal(obj->pe, "calloc(field_grad->delsq) failed");

    /* Allocate data space on target (or alias) */
 
    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, obj->nf*NVECTOR*nsites*sizeof(double));
      tdpMemcpy(&obj->target->grad, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);

      tdpMalloc((void **) &tmp, obj->nf*nsites*sizeof(double));
      tdpMemcpy(&obj->target->delsq, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);
    }
  }

  if (obj->level == 3) {
    obj->d_ab = (double*) calloc(NSYMM*obj->nf*nsites, sizeof(double));
    if (obj->d_ab == NULL) pe_fatal(obj->pe, "calloc(fieldgrad->d_ab) failed\n");

    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, NSYMM*obj->nf*nsites*sizeof(double));
      tdpMemcpy(&obj->target->d_ab, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);
    }
  }

  if (obj->level >= 4) {
    obj->grad_delsq = (double*) calloc(NVECTOR*obj->nf*nsites, sizeof(double));
    obj->delsq_delsq = (double*) calloc(obj->nf*nsites, sizeof(double));
    if (obj->grad_delsq == NULL) pe_fatal(obj->pe, "calloc(grad->grad_delsq) failed");
    if (obj->delsq_delsq == NULL) pe_fatal(obj->pe, "calloc(grad->delsq_delsq) failed");

    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, NVECTOR*obj->nf*nsites*sizeof(double));
      tdpMemcpy(&obj->target->grad_delsq, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice); 

      tdpMalloc((void **) &tmp, obj->nf*nsites*sizeof(double));
      tdpMemcpy(&obj->target->delsq_delsq, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice); 
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_memcpy
 *
 *  ONLY grad and delsq at the moment.
 *
 *****************************************************************************/

__host__ int field_grad_memcpy(field_grad_t * obj, int flag) {

  int ndevice;
  size_t nsz;
  double * tmp = NULL;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {

    nsz = obj->nf*obj->nsite*sizeof(double);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int),
		tdpMemcpyHostToDevice);
      tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
		tdpMemcpyHostToDevice);

      tdpMemcpy(&tmp, &obj->target->grad, sizeof(double *),
		tdpMemcpyDeviceToHost);
      tdpMemcpy(tmp, obj->grad, NVECTOR*nsz, tdpMemcpyHostToDevice);
      tdpMemcpy(&tmp, &obj->target->delsq, sizeof(double *),
		tdpMemcpyDeviceToHost);
      tdpMemcpy(tmp, obj->delsq, nsz, tdpMemcpyHostToDevice);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(&tmp, &obj->target->grad, sizeof(double *),
		tdpMemcpyDeviceToHost);
      tdpMemcpy(obj->grad, tmp, NVECTOR*nsz, tdpMemcpyDeviceToHost);
      tdpMemcpy(&tmp, &obj->target->delsq, sizeof(double *),
		tdpMemcpyDeviceToHost);
      tdpMemcpy(obj->delsq, tmp, nsz, tdpMemcpyDeviceToHost);
      break;
    default:
      pe_fatal(obj->pe, "Bad flag in field_memcpy\n");
      break;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_set
 *
 *****************************************************************************/

int field_grad_set(field_grad_t * obj, grad_ft f2, grad_ft f4) {

  assert(obj);

  obj->d2 = f2;
  obj->d4 = f4;

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_dab_set
 *
 *  The function may be NULL.
 *
 *****************************************************************************/

int field_grad_dab_set(field_grad_t * obj, grad_ft fdab) {

  assert(obj);

  obj->dab = fdab;

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_free
 *
 *  The caller is responsible for releasing the resources associated
 *  with the field itself obj->field.
 *
 *****************************************************************************/

__host__ void field_grad_free(field_grad_t * obj) {

  int ndevice;
  double * tmp;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpMemcpy(&tmp, &obj->target->grad, sizeof(double *),
	      tdpMemcpyDeviceToHost); 
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->delsq, sizeof(double *),
	      tdpMemcpyDeviceToHost); 
    tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->d_ab, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    if (tmp) tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->grad_delsq, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    if (tmp) tdpFree(tmp);
    tdpMemcpy(&tmp, &obj->target->delsq_delsq, sizeof(double *),
	      tdpMemcpyDeviceToHost);
    if (tmp) tdpFree(tmp);

    tdpFree(obj->target);
  }

  if (obj->grad) free(obj->grad);
  if (obj->delsq) free(obj->delsq);
  if (obj->grad_delsq) free(obj->grad_delsq);
  if (obj->delsq_delsq) free(obj->delsq_delsq);
  if (obj->d_ab) free(obj->d_ab);

  obj->field = NULL;
  free(obj);

  return;
}

/*****************************************************************************
 *
 *  field_grad_compute
 *
 *****************************************************************************/

int field_grad_compute(field_grad_t * obj) {

  assert(obj);
  assert(obj->d2);

  field_leesedwards(obj->field);

  obj->d2(obj);

  if (obj->level == 3) {
    assert(obj->dab);
    obj->dab(obj);
  }

  if (obj->level >= 4) {
    assert(obj->d4);
    obj->d4(obj);
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_grad
 *
 *****************************************************************************/

int field_grad_scalar_grad(field_grad_t * obj, int index, double grad[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 1);
  assert(grad);

  for (ia = 0; ia < NVECTOR; ia++) {
    grad[ia] = obj->grad[addr_rank2(obj->nsite, 1, NVECTOR, index, 0, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq) {

  assert(obj);
  assert(obj->nf == 1);
  assert(delsq);

  *delsq = obj->delsq[addr_rank1(obj->nsite, 1, index, 0)];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_grad_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_scalar_grad_delsq(field_grad_t * obj, int index,
				 double grad[3]) {
  int ia;

  assert(obj);
  assert(obj->nf == 1);
  assert(grad);

  for (ia = 0; ia < NVECTOR; ia++) {
    grad[ia] = obj->grad_delsq[addr_rank1(obj->nsite, NVECTOR, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_delsq_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd) {

  assert(obj);
  assert(obj->nf == 1);
  assert(dd);

  *dd = obj->delsq_delsq[addr_rank1(obj->nsite, 1, index, 0)];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_dab
 *
 *  Return tensor d_a d_b for scalar field
 *
 *****************************************************************************/

__host__ __device__
int field_grad_scalar_dab(field_grad_t * obj, int index, double dab[3][3]) {

  assert(obj);
  assert(obj->nf == 1);

  dab[X][X] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, XX)];
  dab[X][Y] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, XY)];
  dab[X][Z] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, XZ)];
  dab[Y][X] = dab[X][Y];
  dab[Y][Y] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, YY)];
  dab[Y][Z] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, YZ)];
  dab[Z][X] = dab[X][Z];
  dab[Z][Y] = dab[Y][Z];
  dab[Z][Z] = obj->d_ab[addr_rank1(obj->nsite, NSYMM, index, ZZ)];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_vector_grad
 *
 *  Return the gradient tensor for vector field dp[ia][ib] = d_a p_b.
 *
 *****************************************************************************/

__host__ __device__
int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]) {

  int ia, ib;

  assert(obj);
  assert(obj->nf == NVECTOR);
  assert(dp);

  for (ia = 0; ia < NVECTOR; ia++) {
    for (ib = 0; ib < obj->nf; ib++) {
      dp[ia][ib] = obj->grad[addr_rank2(obj->nsite, obj->nf, NVECTOR, index, ib, ia)];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_vector_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_vector_delsq(field_grad_t * obj, int index, double delsq[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == NVECTOR);
  assert(delsq);

  for (ia = 0; ia < NVECTOR; ia++) {
    delsq[ia] = obj->delsq[addr_rank1(obj->nsite, NVECTOR, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_tensor_grad
 *
 *  Expand and return the rank 3 gradient of rank 2 tensor q.
 *
 *****************************************************************************/

__host__ __device__
int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]) {

  int ia;

  assert(obj);
  assert(obj->nf == NQAB);
  assert(dq);

  for (ia = 0; ia < NVECTOR; ia++) {
    dq[ia][X][X] = obj->grad[addr_rank2(obj->nsite, NQAB, 3, index, XX, ia)];
    dq[ia][X][Y] = obj->grad[addr_rank2(obj->nsite, NQAB, 3, index, XY, ia)];
    dq[ia][X][Z] = obj->grad[addr_rank2(obj->nsite, NQAB, 3, index, XZ, ia)];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = obj->grad[addr_rank2(obj->nsite, NQAB, 3, index, YY, ia)];
    dq[ia][Y][Z] = obj->grad[addr_rank2(obj->nsite, NQAB, 3, index, YZ, ia)];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_tensor_delsq
 *
 *  Expand and return Laplacian for tensor field.
 *
 *****************************************************************************/

__host__ __device__
int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(dsq);

  dsq[X][X] = obj->delsq[addr_rank1(obj->nsite, NQAB, index, XX)];
  dsq[X][Y] = obj->delsq[addr_rank1(obj->nsite, NQAB, index, XY)];
  dsq[X][Z] = obj->delsq[addr_rank1(obj->nsite, NQAB, index, XZ)];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = obj->delsq[addr_rank1(obj->nsite, NQAB, index, YY)];
  dsq[Y][Z] = obj->delsq[addr_rank1(obj->nsite, NQAB, index, YZ)];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];

  return 0;
}
*****************************************************************************
*
*  field_grad_array_grad
*
*  Return the gradient tensor for array field dp[ia][ib] = d_a p_b.
*
*****************************************************************************/

__host__ __device__
int field_grad_array_grad(field_grad_t * obj, int index, double *dp) {
    
    int ia, ib;
    
    assert(obj);
    assert(dp);
    
    for (ia = 0; ia < obj->nf; ia++) {
        for (ib = 0; ib < obj->nf; ib++) {
            dp[ia][ib] = obj->grad[addr_rank2(obj->nsite, obj->nf, obj->nf, index, ib, ia)];
        }
    }
    
    return 0;
}

/*****************************************************************************
 *
 *  field_grad_array_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_array_delsq(field_grad_t * obj, int index, double *delsq) {
    
    int ia;
    
    assert(obj);
    assert(delsq);
    
    for (ia = 0; ia < obj->nf; ia++) {
        delsq[ia] = obj->delsq[addr_rank1(obj->nsite, obj->nf, index, ia)];
    }
    
    return 0;
}
