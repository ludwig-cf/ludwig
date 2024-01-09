/*****************************************************************************
 *
 *  field_grad.c
 *
 *  Gradients (not just "grad" in the mathematical sense) of a field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_grad.h"

static int field_grad_init(field_grad_t * obj);

/*****************************************************************************
 *
 *  field_grad_create
 *
 *****************************************************************************/

__host__ int field_grad_create(pe_t * pe, field_t * f, int level,
			       field_grad_t ** pobj) {

  int ifail = 0;
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

  ifail = field_grad_init(obj);
  if (ifail != 0) {
    pe_info(pe, "field_grad: failure in int32_t indexing\n");
    pe_fatal(pe, "The local doamin size is too large\n");
  }

  *pobj = obj;

  return ifail;
}

/*****************************************************************************
 *
 *  field_grad_init
 *
 *****************************************************************************/

static int field_grad_init(field_grad_t * obj) {

  int ndevice;
  int nsites;
  size_t nfsz;
  double * tmp;

  assert(obj);

  nsites = obj->field->nsites;
  obj->nsite = nsites;
  nfsz = (size_t) obj->nf*nsites;

  /* Failure in int32_t indexing ... */
  if (INT_MAX < nfsz || nfsz < 1) return -1;

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
    if (INT_MAX/NVECTOR < nfsz) return -1;
    obj->grad  = (double *) calloc(NVECTOR*nfsz, sizeof(double));
    obj->delsq = (double *) calloc(nfsz, sizeof(double));

    if (obj->grad == NULL) pe_fatal(obj->pe, "calloc(field_grad->grad) failed");
    if (obj->delsq == NULL) pe_fatal(obj->pe, "calloc(field_grad->delsq) failed");

    /* Allocate data space on target (or alias) */
 
    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, nfsz*NVECTOR*sizeof(double));
      tdpMemcpy(&obj->target->grad, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);

      tdpMalloc((void **) &tmp, nfsz*sizeof(double));
      tdpMemcpy(&obj->target->delsq, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);
    }
  }

  if (obj->level == 3) {
    if (INT_MAX/NSYMM < nfsz) return -1;
    obj->d_ab = (double*) calloc(NSYMM*nfsz, sizeof(double));
    if (obj->d_ab == NULL) pe_fatal(obj->pe, "calloc(fieldgrad->d_ab) failed\n");

    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, NSYMM*nfsz*sizeof(double));
      tdpMemcpy(&obj->target->d_ab, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice);
    }
  }

  if (obj->level >= 4) {
    obj->grad_delsq = (double*) calloc(NVECTOR*nfsz, sizeof(double));
    obj->delsq_delsq = (double*) calloc(nfsz, sizeof(double));
    if (obj->grad_delsq == NULL) pe_fatal(obj->pe, "calloc(grad->grad_delsq)\n");
    if (obj->delsq_delsq == NULL) pe_fatal(obj->pe, "calloc(grad->delsq_delsq) failed");

    if (ndevice > 0) {
      tdpMalloc((void **) &tmp, NVECTOR*nfsz*sizeof(double));
      tdpMemcpy(&obj->target->grad_delsq, &tmp, sizeof(double *),
		tdpMemcpyHostToDevice); 

      tdpMalloc((void **) &tmp, nfsz*sizeof(double));
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

__host__ int field_grad_memcpy(field_grad_t * obj, tdpMemcpyKind flag) {

  int ndevice;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {

    size_t nsz = (size_t) obj->nf*obj->nsite*sizeof(double);
    double * grad = NULL;
    double * delsq = NULL;
    double * grad_delsq = NULL;
    double * delsq_delsq = NULL;

    tdpAssert(tdpMemcpy(&grad, &obj->target->grad, sizeof(double *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(&delsq, &obj->target->delsq, sizeof(double *),
			tdpMemcpyDeviceToHost));
    if (obj->level >= 4) {
      tdpAssert(tdpMemcpy(&grad_delsq, &obj->target->grad_delsq,
			  sizeof(double *), tdpMemcpyDeviceToHost));
      tdpAssert(tdpMemcpy(&delsq_delsq, &obj->target->delsq_delsq,
			  sizeof(double *), tdpMemcpyDeviceToHost));
    }

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&obj->target->nf, &obj->nf, sizeof(int),
		tdpMemcpyHostToDevice);
      tdpMemcpy(&obj->target->nsite, &obj->nsite, sizeof(int),
		tdpMemcpyHostToDevice);

      tdpMemcpy(grad, obj->grad, NVECTOR*nsz, tdpMemcpyHostToDevice);
      tdpMemcpy(delsq, obj->delsq, nsz, tdpMemcpyHostToDevice);
      if (obj->level >= 4) {
	tdpMemcpy(grad_delsq, obj->grad_delsq, NVECTOR*nsz, flag);
	tdpMemcpy(delsq_delsq, obj->delsq_delsq, nsz, flag);
      }
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(obj->grad, grad, NVECTOR*nsz, tdpMemcpyDeviceToHost);
      tdpMemcpy(obj->delsq, delsq, nsz, tdpMemcpyDeviceToHost);
      if (obj->level >= 4) {
	tdpAssert(tdpMemcpy(obj->grad_delsq, grad_delsq, nsz*NVECTOR, flag));
	tdpAssert(tdpMemcpy(obj->delsq_delsq, delsq_delsq, nsz, flag));
      }
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

__host__ __device__
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
 *  field_grad_pair_grad
 *
 *****************************************************************************/

__host__ __device__
int field_grad_pair_grad(field_grad_t * obj, int index, double grad[2][3]) {

  int ia;

  assert(obj);
  assert(obj->nf == 2);

  for (ia = 0; ia < NVECTOR; ia++) {
    grad[0][ia] = obj->grad[addr_rank2(obj->nsite, 2, NVECTOR, index, 0, ia)];
  }
  for (ia = 0; ia < NVECTOR; ia++) {
    grad[1][ia] = obj->grad[addr_rank2(obj->nsite, 2, NVECTOR, index, 1, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_pair_delsq
 *
 *****************************************************************************/

__host__ __device__
int field_grad_pair_delsq(field_grad_t * obj, int index, double * delsq) {

  int ia;

  assert(obj);
  assert(obj->nf == 2);

  for (ia = 0; ia < obj->nf; ia++) {
    delsq[ia] = obj->delsq[addr_rank1(obj->nsite, obj->nf, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_pair_grad_set
 *
 *****************************************************************************/

__host__ __device__
int field_grad_pair_grad_set(field_grad_t * obj, int index,
			     const double grad[2][3]) {
  int ia;

  assert(obj);
  assert(obj->nf == 2);

  for (ia = 0; ia < NVECTOR; ia++) {
    obj->grad[addr_rank2(obj->nsite, 2, NVECTOR, index, 0, ia)] = grad[0][ia];
  }
  for (ia = 0; ia < NVECTOR; ia++) {
    obj->grad[addr_rank2(obj->nsite, 2, NVECTOR, index, 1, ia)] = grad[1][ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_pair_delsq_set
 *
 *****************************************************************************/

__host__ __device__
int field_grad_pair_delsq_set(field_grad_t * obj, int index,
			      const double * delsq) {
  int ia;

  assert(obj);
  assert(obj->nf == 2);

  for (ia = 0; ia < obj->nf; ia++) {
    obj->delsq[addr_rank1(obj->nsite, obj->nf, index, ia)] = delsq[ia];
  }

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
