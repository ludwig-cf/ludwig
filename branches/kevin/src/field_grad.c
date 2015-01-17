/*****************************************************************************
 *
 *  field_grad.c
 *
 *  Gradients (not just "grad" in the mathematical sense) of a field.
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

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "targetDP.h"

static int field_grad_init(field_grad_t * obj);

/*****************************************************************************
 *
 *  field_grad_create
 *
 *****************************************************************************/

int field_grad_create(field_t * f, int level, field_grad_t ** pobj) {

  field_grad_t * obj =  (field_grad_t *) NULL;

  assert(f);
  assert(pobj);

  obj = (field_grad_t *) calloc(1, sizeof(field_grad_t));
  if (obj == NULL) fatal("calloc(field_grad_t) failed\n");

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

  int nsites;

  assert(obj);
  
  nsites = obj->field->nsites;

  if (obj->level >= 2) {
    obj->grad = (double *) calloc(NVECTOR*obj->nf*nsites, sizeof(double));
    obj->delsq = (double *) calloc(obj->nf*nsites, sizeof(double));
    if (obj->grad == NULL) fatal("calloc(field_grad->grad) failed");
    if (obj->delsq == NULL) fatal("calloc(field_grad->delsq) failed");

    /* allocate target copies */
    targetCalloc((void **) &obj->t_grad, NVECTOR*obj->nf*nsites*sizeof(double));
    targetCalloc((void **) &obj->t_delsq, obj->nf*nsites*sizeof(double));


  }

  if (obj->level == 3) {
    obj->d_ab = (double*) calloc(NSYMM*obj->nf*nsites, sizeof(double));
    if (obj->d_ab == NULL) fatal("calloc(fieldgrad->d_ab) failed\n");
  }

  if (obj->level >= 4) {
    obj->grad_delsq = (double*) calloc(NVECTOR*obj->nf*nsites, sizeof(double));
    obj->delsq_delsq = (double*) calloc(obj->nf*nsites, sizeof(double));
    if (obj->grad_delsq == NULL) fatal("calloc(grad->grad_delsq) failed");
    if (obj->delsq_delsq == NULL) fatal("calloc(grad->delsq_delsq) failed");
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

int field_grad_dab_set(field_grad_t * obj, dab_ft fdab) {

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

void field_grad_free(field_grad_t * obj) {

  assert(obj);

  if (obj->grad) free(obj->grad);
  if (obj->delsq) free(obj->delsq);
  if (obj->t_grad) targetFree(obj->t_grad);
  if (obj->t_delsq) targetFree(obj->t_delsq);
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

  //  obj->d2(obj->field->nf, obj->field->data, obj->grad, obj->delsq);

  obj->d2(obj->field->nf, obj->field->data,obj->field->t_data, obj->grad, obj->t_grad, obj->delsq, obj->t_delsq, obj->field->siteMask, obj->field->t_siteMask);

  if (obj->level == 3) {
    assert(obj->dab);
    obj->dab(obj->field->nf, obj->field->data, obj->d_ab);
  }

  if (obj->level >= 4) {
    assert(obj->d4);
    // obj->d4(obj->field->nf, obj->delsq, obj->grad_delsq, obj->delsq_delsq);
    //obj->d4(obj->field->nf, obj->field->data,obj->field->t_data, obj->grad, obj->t_grad, obj->delsq, obj->t_delsq, obj->field->siteMask, obj->field->t_siteMask);

    //TO DO TEMPORARY FIX to allow free energies which require del^4 phi - 
    // we are just using the existing t_* data structures 
    // for the higher order host structures - this needs properly sorted

    obj->d4(obj->field->nf, obj->delsq,obj->field->t_data,obj->grad_delsq, obj->t_grad, obj->delsq_delsq, obj->t_delsq, obj->field->siteMask, obj->field->t_siteMask);
    
    //void* dummy;
    //obj->d4(obj->field->nf, obj->delsq,dummy,obj->grad_delsq, dummy, obj->delsq_delsq, dummy, obj->field->siteMask, obj->field->t_siteMask);

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

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = obj->grad[NVECTOR*index + ia];
  }
 
  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_delsq
 *
 *****************************************************************************/

int field_grad_scalar_delsq(field_grad_t * obj, int index, double * delsq) {

  assert(obj);
  assert(obj->nf == 1);
  assert(delsq);

  *delsq = obj->delsq[index];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_grad_delsq
 *
 *****************************************************************************/

int field_grad_scalar_grad_delsq(field_grad_t * obj, int index,
				 double grad[3]) {
  int ia;

  assert(obj);
  assert(obj->nf == 1);
  assert(grad);

  for (ia = 0; ia < 3; ia++) {
    grad[ia] = obj->grad_delsq[NVECTOR*index + ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_delsq_delsq
 *
 *****************************************************************************/

int field_grad_scalar_delsq_delsq(field_grad_t * obj, int index, double * dd) {

  assert(obj);
  assert(obj->nf == 1);
  assert(dd);

  *dd = obj->delsq_delsq[index];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_scalar_dab
 *
 *  Return tensor d_a d_b for scalar field
 *
 *****************************************************************************/

int field_grad_scalar_dab(field_grad_t * obj, int index, double dab[3][3]) {

  assert(obj);
  assert(obj->nf == 1);

  dab[X][X] = obj->d_ab[NSYMM*index + XX];
  dab[X][Y] = obj->d_ab[NSYMM*index + XY];
  dab[X][Z] = obj->d_ab[NSYMM*index + XZ];
  dab[Y][X] = dab[X][Y];
  dab[Y][Y] = obj->d_ab[NSYMM*index + YY];
  dab[Y][Z] = obj->d_ab[NSYMM*index + YZ];
  dab[Z][X] = dab[X][Z];
  dab[Z][Y] = dab[Y][Z];
  dab[Z][Z] = obj->d_ab[NSYMM*index + ZZ];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_vector_grad
 *
 *  Return the gradient tensor for vector field dp[ia][ib] = d_a p_b.
 *
 *****************************************************************************/

int field_grad_vector_grad(field_grad_t * obj, int index, double dp[3][3]) {

  int ia, ib;

  assert(obj);
  assert(obj->nf == NVECTOR);
  assert(dp);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      dp[ia][ib] = obj->grad[NVECTOR*(obj->nf*index + ib) + ia];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_vector_delsq
 *
 *****************************************************************************/

int field_grad_vector_delsq(field_grad_t * obj, int index, double delsq[3]) {

  int ia;

  assert(obj);
  assert(obj->nf == NVECTOR);
  assert(delsq);

  for (ia = 0; ia < 3; ia++) {
    delsq[ia] = obj->delsq[NVECTOR*index + ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_pair_grad
 *
 *  Expand and return grad[3][2]
 *
 *****************************************************************************/

int field_grad_pair_grad(field_grad_t * obj, int index, double grad[2][3]) {

  assert(obj);
  assert(obj->nf == NPAIR);

  grad[0][X] = obj->grad[NVECTOR*(NPAIR*index + X) + 0];
  grad[1][X] = obj->grad[NVECTOR*(NPAIR*index + X) + 1];
  grad[0][Y] = obj->grad[NVECTOR*(NPAIR*index + Y) + 0];
  grad[1][Y] = obj->grad[NVECTOR*(NPAIR*index + Y) + 1];
  grad[0][Z] = obj->grad[NVECTOR*(NPAIR*index + Z) + 0];
  grad[1][Z] = obj->grad[NVECTOR*(NPAIR*index + Z) + 1];

  return 0;
}

/*****************************************************************************
 *
 * field_grad_pair_delsq
 *
 *****************************************************************************/

int field_grad_pair_delsq(field_grad_t * obj, int index, double delsq[2]) {

  assert(obj);
  assert(obj->nf == NPAIR);

  delsq[0] = obj->delsq[NPAIR*index + 0];
  delsq[1] = obj->delsq[NPAIR*index + 1];

  return 0;
}

/*****************************************************************************
 *
 *  field_grad_tensor_grad
 *
 *  Expand and return the rank 3 gradient of rank 2 tensor q.
 *
 *****************************************************************************/

int field_grad_tensor_grad(field_grad_t * obj, int index, double dq[3][3][3]) {

  int ia;

  assert(obj);
  assert(obj->nf == NQAB);
  assert(dq);

  for (ia = 0; ia < NVECTOR; ia++) {
    dq[ia][X][X] = obj->grad[NVECTOR*(NQAB*index + XX) + ia];
    dq[ia][X][Y] = obj->grad[NVECTOR*(NQAB*index + XY) + ia];
    dq[ia][X][Z] = obj->grad[NVECTOR*(NQAB*index + XZ) + ia];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = obj->grad[NVECTOR*(NQAB*index + YY) + ia];
    dq[ia][Y][Z] = obj->grad[NVECTOR*(NQAB*index + YZ) + ia];
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

int field_grad_tensor_delsq(field_grad_t * obj, int index, double dsq[3][3]) {

  assert(obj);
  assert(obj->nf == NQAB);
  assert(dsq);

  dsq[X][X] = obj->delsq[NQAB*index + XX];
  dsq[X][Y] = obj->delsq[NQAB*index + XY];
  dsq[X][Z] = obj->delsq[NQAB*index + XZ];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = obj->delsq[NQAB*index + YY];
  dsq[Y][Z] = obj->delsq[NQAB*index + YZ];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];

  return 0;
}

