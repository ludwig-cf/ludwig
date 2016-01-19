/*****************************************************************************
 *
 *  test_field_grad.c
 *
 *  This tests the mechanics of the field_grad object rather than
 *  any real implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field.h"
#include "field_grad_s.h"
#include "tests.h"

enum encode {ENCODE_GRAD = 1, ENCODE_DELSQ, ENCODE_GRAD4, ENCODE_DELSQ4,
             ENCODE_DAB};

static int do_test1(void);
static int do_test3(void);
static int do_test5(void);
static int do_test_dab(void);
static int test_d2(int nf, const double * field, 
		   double * t_field,
		   double * grad,
		   double * t_grad,
		   double * delsq,
		   double * t_delsq
		   );

static int test_d4(int nf, const double * field, 
		   double * t_field,
		   double * grad,
		   double * t_grad,
		   double * delsq,
		   double * t_delsq
		   );

static int test_dab(int nf, const double * field, double * dab);
static double test_encode(int code, int nf, int n, int iv);

/*****************************************************************************
 *
 *  test_field_grad_suite
 *
 *****************************************************************************/

int test_field_grad_suite(void) {

  pe_init_quiet();

  /* info("Field gradient object test\n");*/

  do_test1();
  do_test3();
  do_test5();
  do_test_dab();

  info("PASS     ./unit/test_field_grad\n");
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *****************************************************************************/

int do_test1(void) {

  int nfref = 1;
  double delsq;
  double grad[3];

  field_t * field = NULL;
  field_grad_t * gradient = NULL;

  coords_init();
  le_init();

  field_create(nfref, "scalar-field-test", &field);
  assert(field);
  field_init(field, 0);

  field_grad_create(field, 4, &gradient);
  assert(gradient);
  field_grad_set(gradient, test_d2, test_d4);
  field_grad_compute(gradient);

  field_grad_scalar_grad(gradient, 1, grad);
  assert(fabs(grad[X] - test_encode(ENCODE_GRAD, nfref, X, 0)) < DBL_EPSILON);
  assert(fabs(grad[Y] - test_encode(ENCODE_GRAD, nfref, Y, 0)) < DBL_EPSILON);
  assert(fabs(grad[Z] - test_encode(ENCODE_GRAD, nfref, Z, 0)) < DBL_EPSILON);

  field_grad_scalar_delsq(gradient, 1, &delsq);
  assert(fabs(delsq - test_encode(ENCODE_DELSQ, nfref, X, 0)) < DBL_EPSILON);

  field_grad_scalar_grad_delsq(gradient, 1, grad);
  assert(fabs(grad[X] - test_encode(ENCODE_GRAD4, nfref, X, 0)) < DBL_EPSILON);
  assert(fabs(grad[Y] - test_encode(ENCODE_GRAD4, nfref, Y, 0)) < DBL_EPSILON);
  assert(fabs(grad[Z] - test_encode(ENCODE_GRAD4, nfref, Z, 0)) < DBL_EPSILON);

  field_grad_scalar_delsq_delsq(gradient, 1, &delsq);
  assert(fabs(delsq - test_encode(ENCODE_DELSQ4, nfref, X, 0)) < DBL_EPSILON);

  field_grad_free(gradient);
  field_free(field);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test3
 *
 *****************************************************************************/

static int do_test3(void) {

  int nf = 3;
  double delsq[3];
  double grad[3][3];

  field_t * field = NULL;
  field_grad_t * gradient = NULL;

  coords_init();
  le_init();

  field_create(nf, "vector-field-test", &field);
  assert(field);
  field_init(field, 0);

  field_grad_create(field, 4, &gradient);
  assert(gradient);
  field_grad_set(gradient, test_d2, test_d4);
  field_grad_compute(gradient);

  field_grad_vector_grad(gradient, 1, grad);
  assert(fabs(grad[X][X] - test_encode(ENCODE_GRAD, nf, X, X)) < DBL_EPSILON);
  assert(fabs(grad[X][Y] - test_encode(ENCODE_GRAD, nf, X, Y)) < DBL_EPSILON);
  assert(fabs(grad[X][Z] - test_encode(ENCODE_GRAD, nf, X, Z)) < DBL_EPSILON);

  field_grad_vector_delsq(gradient, 1, delsq);
  assert(fabs(delsq[X] - test_encode(ENCODE_DELSQ, nf, X, X)) < DBL_EPSILON);
  assert(fabs(delsq[Y] - test_encode(ENCODE_DELSQ, nf, X, Y)) < DBL_EPSILON);
  assert(fabs(delsq[Z] - test_encode(ENCODE_DELSQ, nf, X, Z)) < DBL_EPSILON);

  field_grad_free(gradient);
  field_free(field);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test5
 *
 *****************************************************************************/

static int do_test5(void) {

  int nf = 5;
  int ia;
  double tol = DBL_EPSILON;
  double delsq[3][3];
  double grad[3][3][3];

  field_t * field = NULL;
  field_grad_t * gradient = NULL;

  coords_init();
  le_init();

  field_create(nf, "tensor-field-test", &field);
  assert(field);
  field_init(field, 0);

  field_grad_create(field, 4, &gradient);
  assert(gradient);
  field_grad_set(gradient, test_d2, test_d4);
  field_grad_compute(gradient);

  field_grad_tensor_grad(gradient, 1, grad);
  for (ia = 0; ia < 3; ia++) {
    assert(fabs(grad[ia][X][X] - test_encode(ENCODE_GRAD, nf, ia, XX)) < tol);
    assert(fabs(grad[ia][X][Y] - test_encode(ENCODE_GRAD, nf, ia, XY)) < tol);
    assert(fabs(grad[ia][X][Z] - test_encode(ENCODE_GRAD, nf, ia, XZ)) < tol);
    assert(fabs(grad[ia][Y][X] - test_encode(ENCODE_GRAD, nf, ia, XY)) < tol);
    assert(fabs(grad[ia][Y][Y] - test_encode(ENCODE_GRAD, nf, ia, YY)) < tol);
    assert(fabs(grad[ia][Y][Z] - test_encode(ENCODE_GRAD, nf, ia, YZ)) < tol);
    assert(fabs(grad[ia][Z][X] - test_encode(ENCODE_GRAD, nf, ia, XZ)) < tol);
    assert(fabs(grad[ia][Z][Y] - test_encode(ENCODE_GRAD, nf, ia, YZ)) < tol);
    assert(fabs(grad[ia][Z][Z] + grad[ia][X][X] + grad[ia][Y][Y]) < tol);
  }

  field_grad_tensor_delsq(gradient, 1, delsq);
  assert(fabs(delsq[X][X] - test_encode(ENCODE_DELSQ, nf, X, XX)) < tol);
  assert(fabs(delsq[X][Y] - test_encode(ENCODE_DELSQ, nf, X, XY)) < tol);
  assert(fabs(delsq[X][Z] - test_encode(ENCODE_DELSQ, nf, X, XZ)) < tol);
  assert(fabs(delsq[Y][X] - test_encode(ENCODE_DELSQ, nf, X, XY)) < tol);
  assert(fabs(delsq[Y][Y] - test_encode(ENCODE_DELSQ, nf, X, YY)) < tol);
  assert(fabs(delsq[Y][Z] - test_encode(ENCODE_DELSQ, nf, X, YZ)) < tol);
  assert(fabs(delsq[Z][X] - test_encode(ENCODE_DELSQ, nf, X, XZ)) < tol);
  assert(fabs(delsq[Z][Y] - test_encode(ENCODE_DELSQ, nf, X, YZ)) < tol);
  assert(fabs(delsq[Z][Z] + delsq[X][X] + delsq[Y][Y]) < tol);

  field_grad_free(gradient);
  field_free(field);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_dab
 *
 *****************************************************************************/

int do_test_dab(void) {

  int nf = 1;
  int index = 1;
  double dab[3][3];

  field_t * field = NULL;
  field_grad_t * gradient = NULL;

  coords_init();
  le_init();

  field_create(nf, "dab-field-test", &field);
  assert(field);
  field_init(field, 0);

  field_grad_create(field, 3, &gradient);
  assert(gradient);

  field_grad_set(gradient, test_d2, NULL);
  field_grad_dab_set(gradient, test_dab);
  field_grad_compute(gradient);

  field_grad_scalar_dab(gradient, index, dab);

  assert(fabs(dab[X][X] - test_encode(ENCODE_DAB, nf, XX, 0)) < DBL_EPSILON);
  assert(fabs(dab[X][Y] - test_encode(ENCODE_DAB, nf, XY, 0)) < DBL_EPSILON);
  assert(fabs(dab[X][Z] - test_encode(ENCODE_DAB, nf, XZ, 0)) < DBL_EPSILON);
  assert(fabs(dab[Y][X] - test_encode(ENCODE_DAB, nf, XY, 0)) < DBL_EPSILON);
  assert(fabs(dab[Y][Y] - test_encode(ENCODE_DAB, nf, YY, 0)) < DBL_EPSILON);
  assert(fabs(dab[Y][Z] - test_encode(ENCODE_DAB, nf, YZ, 0)) < DBL_EPSILON);
  assert(fabs(dab[Z][X] - test_encode(ENCODE_DAB, nf, XZ, 0)) < DBL_EPSILON);
  assert(fabs(dab[Z][Y] - test_encode(ENCODE_DAB, nf, YZ, 0)) < DBL_EPSILON);
  assert(fabs(dab[Z][Z] - test_encode(ENCODE_DAB, nf, ZZ, 0)) < DBL_EPSILON);

  /* Clean up */

  field_grad_free(gradient);
  field_free(field);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_d2
 *
 *****************************************************************************/

static int test_d2(int nf, const double * field, 
		   double * t_field,
		   double * grad,
		   double * t_grad,
		   double * delsq,
		   double * t_delsq
		   ) {

  int n;
  int nsites;
  int index = 1;

  assert(grad);
  assert(delsq);

  nsites = coords_nsites();

#ifndef OLD_SHIT
  for (n = 0; n < nf; n++) {
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, X)] = test_encode(ENCODE_GRAD, nf, X, n);
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, Y)] = test_encode(ENCODE_GRAD, nf, Y, n);
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, Z)] = test_encode(ENCODE_GRAD, nf, Z, n);
    delsq[addr_rank1(nsites, nf, index, n)] = test_encode(ENCODE_DELSQ, nf, X, n);
  }
#else
  for (n = 0; n < nf; n++) {
    grad[NVECTOR*(nf*index + n) + X] = test_encode(ENCODE_GRAD, nf, X, n);
    grad[NVECTOR*(nf*index + n) + Y] = test_encode(ENCODE_GRAD, nf, Y, n);
    grad[NVECTOR*(nf*index + n) + Z] = test_encode(ENCODE_GRAD, nf, Z, n);
    delsq[nf*index + n] = test_encode(ENCODE_DELSQ, nf, X, n);
  }
#endif
  return 0;
}

/*****************************************************************************
 *
 *  test_d4
 *
 *****************************************************************************/

static int test_d4(int nf, const double * field, 
		   double * t_field,
		   double * grad,
		   double * t_grad,
		   double * delsq,
		   double * t_delsq
		   ) {
  int n;
  int nsites;
  int index = 1;

  assert(grad);
  assert(delsq);

  nsites = coords_nsites();

#ifndef OLD_SHIT
  for (n = 0; n < nf; n++) {
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, X)] = test_encode(ENCODE_GRAD4, nf, X, n);
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, Y)] = test_encode(ENCODE_GRAD4, nf, Y, n);
    grad[addr_rank2(nsites, nf, NVECTOR, index, n, Z)] = test_encode(ENCODE_GRAD4, nf, Z, n);
    delsq[addr_rank1(nsites, nf, index, n)] = test_encode(ENCODE_DELSQ4, nf, X, n);
  }
#else
  for (n = 0; n < nf; n++) {
    grad[NVECTOR*(nf*index + n) + X] = test_encode(ENCODE_GRAD4, nf, X, n);
    grad[NVECTOR*(nf*index + n) + Y] = test_encode(ENCODE_GRAD4, nf, Y, n);
    grad[NVECTOR*(nf*index + n) + Z] = test_encode(ENCODE_GRAD4, nf, Z, n);
    delsq[nf*index + n] = test_encode(ENCODE_DELSQ4, nf, X, n);
  }
#endif
  return 0;
}

/*****************************************************************************
 *
 *  test_encode
 *
 *****************************************************************************/

static double test_encode(int code, int nf, int iv, int n) {

  double result;

  result = 1.0*code + 0.1*(n*nf + iv);

  return result;
}

/*****************************************************************************
 *
 *  test_dab
 *
 *****************************************************************************/

static int test_dab(int nf, const double * field, double * dab) {

  int n;
  int nsites;
  int index = 1;

  assert(nf == 1);
  assert(field);
  assert(dab);

  nsites = coords_nsites();

#ifndef OLD_SHIT
  for (n = 0; n < nf; n++) {
    dab[addr_dab(nsites, index, XX)] = test_encode(ENCODE_DAB, nf, XX, n);
    dab[addr_dab(nsites, index, XY)] = test_encode(ENCODE_DAB, nf, XY, n);
    dab[addr_dab(nsites, index, XZ)] = test_encode(ENCODE_DAB, nf, XZ, n);
    dab[addr_dab(nsites, index, YY)] = test_encode(ENCODE_DAB, nf, YY, n);
    dab[addr_dab(nsites, index, YZ)] = test_encode(ENCODE_DAB, nf, YZ, n);
    dab[addr_dab(nsites, index, ZZ)] = test_encode(ENCODE_DAB, nf, ZZ, n);
  }
#else
  for (n = 0; n < nf; n++) {
    dab[NSYMM*(nf*index + n) + XX] = test_encode(ENCODE_DAB, nf, XX, n);
    dab[NSYMM*(nf*index + n) + XY] = test_encode(ENCODE_DAB, nf, XY, n);
    dab[NSYMM*(nf*index + n) + XZ] = test_encode(ENCODE_DAB, nf, XZ, n);
    dab[NSYMM*(nf*index + n) + YY] = test_encode(ENCODE_DAB, nf, YY, n);
    dab[NSYMM*(nf*index + n) + YZ] = test_encode(ENCODE_DAB, nf, YZ, n);
    dab[NSYMM*(nf*index + n) + ZZ] = test_encode(ENCODE_DAB, nf, ZZ, n);
  }
#endif
  return 0;
}
