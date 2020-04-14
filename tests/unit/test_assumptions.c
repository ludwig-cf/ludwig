/*****************************************************************************
 *
 *  test_assumptions.c
 *
 *  Test basic model assumptions, portability issues.
 *
 *  Also testing pe_fenv.h, util.h stuff at the moment.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2019 The University of Edinburgh 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "mpi.h"
#include "pe_fenv.h"
#include "util.h"
#include "tests.h"

void (* p_function)(void);
void test_util(void);
int test_pe_fenv(void);
int test_util_discrete_volume(void);
int test_discrete_volume_sphere(double r0[3], double a0, double answer);
int test_macro_abuse(void);

/*****************************************************************************
 *
 *  test_assumptions_suite
 *
 *****************************************************************************/

int test_assumptions_suite(void) {

  double pi;
  PI_DOUBLE(pi_);

  /*
  printf("Testing assumptions...\n");
  printf("This code compiled at %s on %s\n", __TIME__, __DATE__);
  */
  /* All integers in the code should be declared 'int', which
   * we expect to be (at the least) 4 bytes. */

  /*printf("Checking sizeof(int) is 4 bytes... ");*/
  assert(sizeof(int) == 4);
  /* printf("yes\n");*/

  /* printf("Checking sizeof(long int) is >= 4 bytes... ");*/
  assert(sizeof(long int) >= 4);
  /* printf("yes (%ld bytes)\n", sizeof(long int));*/

  /* All floating point types in the code should be double,
   * which must be 8 bytes. */

  /* printf("Checking sizeof(float) is 4 bytes... ");*/
  assert(sizeof(float) == 4);
  /* printf("yes\n");*/

  /* printf("Checking sizeof(double) is 8 bytes... ");*/
  assert(sizeof(double) == 8);
  /* printf("yes\n");*/
  /*
  printf("sizeof(char) = %lu\n", sizeof(char));
  printf("sizeof(unsigned char) = %lu\n", sizeof(unsigned char));
  printf("Checking sizeof(void *) ...\n");
  printf("%lu\n", sizeof(void *));
  */
  /* printf("Checking FILENAME_MAX >= 128 characters ... ");*/
  assert(FILENAME_MAX >= 128);
  /* printf("yes (%d characters)\n", FILENAME_MAX);
     printf("Checking BUFSIZ... (%d bytes)\n", BUFSIZ);

  printf("Checking sizeof(void *)... (%lu bytes)\n", sizeof(void *));
  printf("Checking sizeof(func *)... (%lu bytes)\n", sizeof(p_function));
  */

  /* Maths? */
  /*
  printf("FLT_EPSILON is %14.7e\n", FLT_EPSILON);
  printf("M_PI        is %14.7e\n", M_PI);
  */
  pi = 4.0*atan(1.0);
  test_assert(fabs(pi - pi_) < DBL_EPSILON);


  test_pe_fenv();
  test_util();
  test_macro_abuse();


  /* Information */
  /* printf("Language\n");
     printf("__STC__ = %d\n", __STDC__);*/
#if (__STDC_VERSION__ >= 199901)
  /* printf("__STDC_VERSION__ = %ld\n", __STDC_VERSION__);*/
#endif
  /*
  printf("__DATE__ is %s\n", __DATE__);
  printf("__TIME__ is %s\n", __TIME__);
  printf("__FILE__ is %s\n", __FILE__);
  printf("__LINE__ is %d\n", __LINE__);

  printf("Host appears to be %s-endian\n", is_bigendian() ? "big" : "little");

  printf("All assumptions ok!\n");
  */

  return 0;
}

/*****************************************************************************
 *
 *  test_util
 *
 *****************************************************************************/

void test_util(void) {

  int i, j, k, m, n;
  int sumd, sume;
  KRONECKER_DELTA_CHAR(d_);
  LEVI_CIVITA_CHAR(e_);
  
  /* Krocker delta and Levi-Civita tensor */

  /* printf("Kroneker delta d_[i][j] correct...");*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (i == j) {
	test_assert(d_[i][j] == 1);
      }
      else {
	test_assert(d_[i][j] == 0);
      }
    }
  }

  /* Do some permutations to test the perutation tensor e_[i][j][k].
   * Also use the identity e_ijk e_imn = d_jm d_kn - d_jn d_km
   * for a test. */ 

  /* printf("Permutation tensor e_[i][j][k] correct...");*/

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	test_assert((e_[i][j][k] + e_[i][k][j]) == 0);
	test_assert((e_[i][j][k] + e_[j][i][k]) == 0);
	test_assert((e_[i][j][k] - e_[k][i][j]) == 0);
      }
    }
  }

  for (j = 0; j < 3; j++) {
    for (k = 0; k < 3; k++) {
      for (m = 0; m < 3; m++) {
	for (n = 0; n < 3; n++) {
	  sume = 0.0;
	  for (i = 0; i < 3; i++) {
	    sume += e_[i][j][k]*e_[i][m][n];
	  }
	  sumd = (d_[j][m]*d_[k][n] - d_[j][n]*d_[k][m]);
	  test_assert((sume - sumd) == 0);
	}
      }
    }
  }

  /* printf("yes.\n");*/

  test_util_discrete_volume();

  return;
}

/*****************************************************************************
 *
 *  test_util_discrete_volume
 *
 *  The test values come from an original program to compute the
 *  varience of the discrete colloid size as a function of radius.
 *
 *****************************************************************************/

int test_util_discrete_volume(void) {

  double r[3];

  r[0] = 0.0; r[1] = 0.0; r[2] = 0.0;
  assert(test_discrete_volume_sphere(r, 1.0, 1.0) == 0);
  r[0] = 0.5; r[1] = 0.5; r[2] = 0.5;
  assert(test_discrete_volume_sphere(r, 1.0, 8.0) == 0);

  r[0] = 0.52; r[1] = 0.10; r[2] = 0.99;
  assert(test_discrete_volume_sphere(r, 1.25, 10.0) == 0);
  r[0] = 0.52; r[1] = 0.25; r[2] = 0.99;
  assert(test_discrete_volume_sphere(r, 1.25,  8.0) == 0);

  r[0] = 0.0; r[1] = 0.0; r[2] = 0.0;
  assert(test_discrete_volume_sphere(r, 2.3, 57.0) == 0);
  r[0] = 0.5; r[1] = 0.5; r[2] = 0.5;
  assert(test_discrete_volume_sphere(r, 2.3, 56.0) == 0);

  r[0] = 0.52; r[1] = 0.10; r[2] = 0.99;
  assert(test_discrete_volume_sphere(r, 4.77, 461.0) == 0);
  r[0] = 0.52; r[1] = 0.25; r[2] = 0.99;
  assert(test_discrete_volume_sphere(r, 4.77, 453.0) == 0);

  /* Keep a non-asserted call to prevent compiler warnings
   * about unused r under -DNDEBUG */
  test_discrete_volume_sphere(r, 4.77, 453.0);

  return 0;
}

/*****************************************************************************
 *
 *  test_discrete_volume_sphere
 *
 *  Position is r0 (can be 0 <= x < 1 etc), a0 is radius, and answer
 *  is the expected result.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

int test_discrete_volume_sphere(double r0[3], double a0, double answer) {

  int ifail = 0;
  double vn;

  util_discrete_volume_sphere(r0, a0, &vn);
  if (fabs(vn - answer) > TEST_DOUBLE_TOLERANCE) ifail++;

  /* Move to +ve coords */
  r0[0] += 100.; r0[1] += 100.0; r0[2] += 100.0;
  util_discrete_volume_sphere(r0, a0, &vn);
  if (fabs(vn - answer) > TEST_DOUBLE_TOLERANCE) ifail++;

  /* Move to -ve coords */
  r0[0] -= 200.; r0[1] -= 200.0; r0[2] -= 200.0;
  util_discrete_volume_sphere(r0, a0, &vn);
  if (fabs(vn - answer) > TEST_DOUBLE_TOLERANCE) ifail++;

  return ifail;
}

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

#define MARGS_(_2, _1, N, ...) N 
#define MARGS(...) MARGS_(__VA_ARGS__, 3arg, 2arg, not_allowed)

#define blah_parallel_3arg(index, ndata, stride) \
  index = (ndata) + (stride)
#define blah_parallel_2arg(index, ndata) blah_parallel_3arg(index, ndata, 1)
#define blah_parallel(index, ...) \
  CONCAT(blah_parallel_, MARGS(__VA_ARGS__))(index, __VA_ARGS__)

int test_macro_abuse(void) {

  int sum;

  blah_parallel(sum, 2);
  test_assert(sum == 3);
  blah_parallel(sum, 4, 99);
  test_assert(sum == 103);

  return 0;
}

/*****************************************************************************
 *
 *  test_pe_fenv
 *
 *****************************************************************************/

int test_pe_fenv(void) {

  const char * s = pe_fegetround_tostring();

  test_assert(s != NULL);
  /* printf("Floating point rounding mode: %s\n", s);*/

  return 0;
}
