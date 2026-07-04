/*****************************************************************************
 *
 *  test_util_random.c
 *
 *  (c) 2026 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "util_random_impl.h"

int test_util_random_uniform(void);

/*****************************************************************************
 *
 *  test_util_random_suite
 *
 *****************************************************************************/

int test_util_random_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_random_uniform();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_random_uniform
 *
 *****************************************************************************/

int test_util_random_uniform(void) {

  int ifail = 0;

  const int a = 1389796;    /* the multiplier */
  const int m = 2147483647; /* the modulus */

  {
    int    seed = 1;
    double u01  = util_random_uniform(&seed);
    if (fabs(u01 - 1.0 * a / m) > DBL_EPSILON) {
      ifail = -1;
    }
    assert(seed == a);
    assert(ifail == 0);
  }

  return ifail;
}
