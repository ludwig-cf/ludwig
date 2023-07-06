/*****************************************************************************
 *
 *  test_colloid.c
 *
 *  Unit test for colloid structure.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "colloid.h"
#include "util_fopen.h"
#include "tests.h"

#define TOLERANCE 1.0e-14

int test_colloid_compare(colloid_state_t * s1, colloid_state_t * s2);
int test_are_equal_scalar_double(double a, double b);
int test_are_equal_vector_double(const double * a, const double * b, int nlen);
void test_colloid_ascii_io(colloid_state_t * s, const char * filename);
void test_colloid_binary_io(colloid_state_t * s, const char * filename);

/*****************************************************************************
 *
 *  test_colloid_suite
 *
 *****************************************************************************/

int test_colloid_suite(void) {

  int rank;
  char filename[FILENAME_MAX];

  colloid_state_t sref = {1, 2, 3, 4, 5, 6, 7, 8, 9,
			  {10, 11}, 12, {13, 14, 15}, {16, 17, 18}, 19,
			  {20, 21, 22, 23, 24,
			   25, 26, 27, 28, 29, 30, 31, 32},
			  1.0, 2.0,
			  { 3.0,  4.0,  5.0},
			  { 6.0,  7.0,  8.0},
			  { 9.0, 10.0, 11.0},
			  {12.0, 13.0, 14.0},
			  {15.0, 16.0, 17.0},
			  18.0, -19.0, 20.0, 21.0,
			  {22.0, 23.0, 24.0},
			  25.0, 26.00, 27.0, 28.0,
                          29.0, 30.0, 31.0, 32.0, 33.0, {34.0, 35.0, 36.0},
			  {37.0, 38.0, 39.0, 40.0}, {41.0, 42.0, 43.0, 44.0},
			  {45.0, 46.0, 47.0}};

  const char * tmp_ascii = "/tmp/temp-test-io-file-ascii";
  const char * tmp_binary = "/tmp/temp-test-io-file-binary";

  assert(tmp_ascii);
  assert(tmp_binary);

  /* I assert that the colloid struct is 512 bytes. I.e., don't
   * change it without sorting out the padding. */
  test_assert(sizeof(colloid_state_t) == 512);
  assert(NPAD_INT == 13);
  assert(NPAD_DBL ==  4);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  sprintf(filename, "%s-%3.3d", tmp_ascii, rank);
  test_colloid_ascii_io(&sref, filename);
  remove(filename);

  sprintf(filename, "%s-%3.3d", tmp_binary, rank);
  test_colloid_binary_io(&sref, filename);
  remove(filename);
  
  if (rank == 0) printf("PASS     ./unit/test_colloid\n");

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_ascii_io
 *
 *****************************************************************************/

void test_colloid_ascii_io(colloid_state_t * sref, const char * filename) {

  int n;
  colloid_state_t s = {0};
  FILE * fp = NULL;

  assert(sref);

  fp = util_fopen(filename, "w");

  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_write_ascii(sref, fp);
    fclose(fp);
    test_assert(n == 0);
  }

  fp = NULL;
  fp = util_fopen(filename, "r");

  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_ascii(&s, fp);
    fclose(fp);
    test_assert(n == 0);
  }

  test_colloid_compare(&s, sref);

  return;
}

/*****************************************************************************
 *
 *  test_colloid_binary_io
 *
 ****************************************************************************/

void test_colloid_binary_io(colloid_state_t * sref, const char * filename) {

  int n;
  colloid_state_t s = {0};
  FILE * fp = NULL;

  assert(sref);

  fp = util_fopen(filename, "w");
  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_write_binary(sref, fp);
    fclose(fp);
    test_assert(n == 0);
  }

  fp = NULL;
  fp = util_fopen(filename, "r");
  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_binary(&s, fp);
    fclose(fp);
    test_assert(s.rebuild == 1);
    test_assert(n == 0);
  }

  test_colloid_compare(&s, sref);

  return;
}

/*****************************************************************************
 *
 *  test_colloid_compare
 *
 *  Compare all the elements of two colloid structs (bar rebuild).
 *
 *****************************************************************************/

int test_colloid_compare(colloid_state_t * s1, colloid_state_t * s2) {

  int n;

  test_assert(s1->index    == s2->index);
  test_assert(s1->nbonds   == s2->nbonds);
  test_assert(s1->nangles  == s2->nangles);
  test_assert(s1->isfixedr == s2->isfixedr);
  test_assert(s1->isfixedv == s2->isfixedv);
  test_assert(s1->isfixedw == s2->isfixedw);
  test_assert(s1->isfixeds == s2->isfixeds);
  test_assert(s1->type     == s2->type);

  for (n = 0; n < NBOND_MAX; n++) {
    test_assert(s1->bond[n] == s2->bond[n]);
  }

  assert(s1->rng            == s2->rng);
  assert(s1->isfixedrxyz[0] == s2->isfixedrxyz[0]);
  assert(s1->isfixedrxyz[1] == s2->isfixedrxyz[1]);
  assert(s1->isfixedrxyz[2] == s2->isfixedrxyz[2]);
  assert(s1->isfixedvxyz[0] == s2->isfixedvxyz[0]);
  assert(s1->isfixedvxyz[1] == s2->isfixedvxyz[1]);
  assert(s1->isfixedvxyz[2] == s2->isfixedvxyz[2]);

  test_assert(test_are_equal_scalar_double(s1->a0, s2->a0));
  test_assert(test_are_equal_scalar_double(s1->ah, s2->ah));
  test_assert(test_are_equal_vector_double(s1->r,  s2->r, 3));
  test_assert(test_are_equal_vector_double(s1->v,  s2->v, 3));
  test_assert(test_are_equal_vector_double(s1->w,  s2->w, 3));
  test_assert(test_are_equal_vector_double(s1->s,  s2->s, 3));
  test_assert(test_are_equal_vector_double(s1->m,  s2->m, 3));
  test_assert(test_are_equal_scalar_double(s1->b1, s2->b1));
  test_assert(test_are_equal_scalar_double(s1->b2, s2->b2));
  test_assert(test_are_equal_scalar_double(s1->c,  s2->c));
  test_assert(test_are_equal_scalar_double(s1->h,  s2->h));
  test_assert(test_are_equal_vector_double(s1->dr, s2->dr, 3));

  assert(fabs(s1->deltaphi     - s2->deltaphi) < DBL_EPSILON);
  assert(fabs(s1->q0           - s2->q0)       < DBL_EPSILON);
  assert(fabs(s1->q1           - s2->q1)       < DBL_EPSILON);
  assert(fabs(s1->epsilon      - s2->epsilon)  < DBL_EPSILON);
  assert(fabs(s1->deltaq0      - s2->deltaq0)  < DBL_EPSILON);
  assert(fabs(s1->deltaq1      - s2->deltaq1)  < DBL_EPSILON);
  assert(fabs(s1->sa           - s2->sa)       < DBL_EPSILON);
  assert(fabs(s1->saf          - s2->saf)      < DBL_EPSILON);
  assert(fabs(s1->al           - s2->al)       < DBL_EPSILON);

  assert(fabs(s1->elabc[0]     - s2->elabc[0]) < DBL_EPSILON);
  assert(fabs(s1->elabc[1]     - s2->elabc[1]) < DBL_EPSILON);
  assert(fabs(s1->elabc[2]     - s2->elabc[2]) < DBL_EPSILON);
  assert(fabs(s1->quater[0]    - s2->quater[0]) < DBL_EPSILON);
  assert(fabs(s1->quater[1]    - s2->quater[1]) < DBL_EPSILON);
  assert(fabs(s1->quater[2]    - s2->quater[2]) < DBL_EPSILON);
  assert(fabs(s1->quater[3]    - s2->quater[3]) < DBL_EPSILON);
  assert(fabs(s1->quaterold[0] - s2->quaterold[0]) < DBL_EPSILON);
  assert(fabs(s1->quaterold[1] - s2->quaterold[1]) < DBL_EPSILON);
  assert(fabs(s1->quaterold[2] - s2->quaterold[2]) < DBL_EPSILON);
  assert(fabs(s1->quaterold[3] - s2->quaterold[3]) < DBL_EPSILON);

  /* check the last element of the padding */
  test_assert(test_are_equal_scalar_double(s1->dpad[NPAD_DBL-1], s2->dpad[NPAD_DBL-1]));

  return 0;
}

/*****************************************************************************
 *
 *  test_are_equal_scalar_double
 *
 *****************************************************************************/

int test_are_equal_scalar_double(double a, double b) {

  int iequal = 1;

  if (fabs(a - b) > TOLERANCE) iequal = 0;

  return iequal;
}

/*****************************************************************************
 *
 *  test_are_equal_vector_double
 *
 *****************************************************************************/

int test_are_equal_vector_double(const double * a, const double * b,
				 int nlen) {
  int iequal = 1;
  int n;

  for (n = 0; n < nlen; n++) {
    if (fabs(a[n] - b[n]) > TOLERANCE) iequal = 0;
  }

  return iequal;
}
