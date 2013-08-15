/*****************************************************************************
 *
 *  test_colloid.c
 *
 *  Unit test for colloid structure.
 *
 *  $Id: test_colloid.c,v 1.2 2010-11-02 17:51:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "colloid.h"

#define TOLERANCE 1.0e-14

int test_colloid_compare(colloid_state_t s1, colloid_state_t s2);
int test_are_equal_scalar_double(double a, double b);
int test_are_equal_vector_double(const double * a, const double * b, int nlen);
void test_colloid_ascii_io(colloid_state_t s, const char * filename);
void test_colloid_binary_io(colloid_state_t s, const char * filename);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  colloid_state_t sref = {1, 3, 2, 4, 5, 6, 7, 8, 9,
			  {10, 11},
			  {12, 13, 14, 15, 16,
			   17, 18, 19, 20, 21, 22, 23, 24,
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
                          29.0, 30.0, 31.0, 32.0, {33.0, 34.0, 35.0, 36.0,
			   37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
			   45.0, 46.0, 47.0, 48.0}};

  char * tmp_ascii = NULL;
  char * tmp_binary = NULL;

  /* Use a unique temporary file to prevent i/o collisions if this
   * serial code is started in parallel. */

  tmp_ascii = tmpnam(NULL);
  tmp_binary = tmpnam(NULL);

  assert(tmp_ascii);
  assert(tmp_binary);

  printf("sizeof(colloid_state_t) = %ld\n", sizeof(colloid_state_t));

  /* I assert that the colloid struct is 512 bytes. I.e., don't
   * change it without sorting out the padding. */
  assert(sizeof(colloid_state_t) == 512);

  test_colloid_ascii_io(sref, tmp_ascii);
  test_colloid_binary_io(sref, tmp_binary);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_ascii_io
 *
 *****************************************************************************/

void test_colloid_ascii_io(colloid_state_t sref, const char * filename) {

  int n;
  colloid_state_t s;
  FILE * fp;

  fp = fopen(filename, "w");

  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_write_ascii(sref, fp);
    fclose(fp);
    printf("wrote ref ascii item to %s\n", filename);
    assert(n == 0);
  }

  fp = fopen(filename, "r");

  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_ascii(&s, fp);
    fclose(fp);
    printf("read ref ascii item from %s\n", filename);
    assert(n == 0);
  }

  test_colloid_compare(s, sref);
  printf("ascii write/read correct\n");

  return;
}

/*****************************************************************************
 *
 *  test_colloid_binary_io
 *
 ****************************************************************************/

void test_colloid_binary_io(colloid_state_t sref, const char * filename) {

  int n;
  colloid_state_t s;
  FILE * fp;

  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_write_binary(sref, fp);
    fclose(fp);
    printf("wrote ref binary item to %s\n", filename);
    assert(n == 0);
  }

  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_binary(&s, fp);
    fclose(fp);
    assert(s.rebuild == 1);
    printf("read binary item from %s\n", filename);
    assert(n == 0);
  }

  test_colloid_compare(s, sref);
  printf("binary write/read correct\n");

  return;
}

/*****************************************************************************
 *
 *  test_colloid_compare
 *
 *  Compare all the elements of two colloid structs (bar rebuild).
 *
 *****************************************************************************/

int test_colloid_compare(colloid_state_t s1, colloid_state_t s2) {

  int n;

  assert(s1.index == s2.index);
  assert(s1.nbonds == s2.nbonds);
  assert(s1.nangles == s2.nangles);
  assert(s1.isfixedr == s2.isfixedr);
  assert(s1.isfixedv == s2.isfixedv);
  assert(s1.isfixedw == s2.isfixedw);
  assert(s1.isfixeds == s2.isfixeds);
  assert(s1.type == s2.type);

  for (n = 0; n < NBOND_MAX; n++) {
    assert(s1.bond[n] == s2.bond[n]);
  }

  assert(test_are_equal_scalar_double(s1.a0, s2.a0));
  assert(test_are_equal_scalar_double(s1.ah, s2.ah));
  assert(test_are_equal_vector_double(s1.r, s2.r, 3));
  assert(test_are_equal_vector_double(s1.v, s2.v, 3));
  assert(test_are_equal_vector_double(s1.w, s2.w, 3));
  assert(test_are_equal_vector_double(s1.s, s2.s, 3));
  assert(test_are_equal_vector_double(s1.m, s2.m, 3));
  assert(test_are_equal_scalar_double(s1.b1, s2.b1));
  assert(test_are_equal_scalar_double(s1.b2, s2.b2));
  assert(test_are_equal_scalar_double(s1.c, s2.c));
  assert(test_are_equal_scalar_double(s1.h, s2.h));
  assert(test_are_equal_vector_double(s1.dr, s2.dr, 3));

  /* check the last element of the padding */
  assert(test_are_equal_scalar_double(s1.dpad[NPAD_DBL-1], s2.dpad[NPAD_DBL-1]));

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
