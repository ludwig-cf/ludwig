/*****************************************************************************
 *
 *  test_colloid.c
 *
 *  Unit test for colloid structure.
 *
 *  $Id: test_colloid.c,v 1.1.2.1 2010-06-06 11:44:30 kevin Exp $
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

  colloid_state_t sref = {1, 2, 2.0, 3.0,
			  {99.0, 98.0, 97.0},
			  { 5.0,  6.0,  7.0},
			  { 8.0,  9.0, 10.0},
			  {11.0, 12.0, 13.0},
			  {14.0, 15.0, 16.0},
			  17.0, -18.0, 2.5, 3.5,
			  {19.0, 20.0, 21.0},
			  22.0,
			  0.00,
			  {23.0, 24.0, -25.0}};

  const char * file_ascii = "/tmp/colloid_ascii.dat";
  const char * file_binary = "/tmp/colloid_binary.dat";

  printf("sizeof(colloid_state_t) = %ld\n", sizeof(colloid_state_t));

  test_colloid_ascii_io(sref, file_ascii);
  test_colloid_binary_io(sref, file_binary);

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
    printf("wrote %d ascii item to %s\n", n, filename);
    assert(n == 1);
  }

  fp = fopen(filename, "r");

  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_ascii(&s, fp);
    fclose(fp);
    printf("read %d ascii item from %s\n", n, filename);
    assert(n == 1);
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
    printf("wrote %d binary item to %s\n", n, filename);
    assert(n == 1);
  }

  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("fopen(%s) failed\n", filename);
  }
  else {
    n = colloid_state_read_binary(&s, fp);
    fclose(fp);
    printf("read %d binary item from %s\n", n, filename);
    assert(n == 1);
  }

  test_colloid_compare(s, sref);
  printf("binary write/read correct\n");

  return;
}

/*****************************************************************************
 *
 *  test_colloid_compare
 *
 *  Compare all the elements of two colloid structs.
 *
 *****************************************************************************/

int test_colloid_compare(colloid_state_t s1, colloid_state_t s2) {

  assert(s1.index == s2.index);
  assert(s1.rebuild == s2.rebuild);
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
