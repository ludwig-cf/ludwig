/*****************************************************************************
 *
 *  test_util_math.c
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "util_math_inline.h"

int test_util_math_imax(void);
int test_util_math_imin(void);
int test_util_math_square_modulus_int8(void);

/*****************************************************************************
 *
 *  test_util_math_suite
 *
 *****************************************************************************/

int test_util_math_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_math_imax();
  test_util_math_imin();
  test_util_math_square_modulus_int8();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_math_imax
 *
 *****************************************************************************/

int test_util_math_imax(void) {

  int ifail = 0;

  ifail = util_imax(0, 1);
  assert(ifail == 1);

  ifail = util_imax(-1, 0);
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_math_imin
 *
 *****************************************************************************/

int test_util_math_imin(void) {

  int ifail = 0;

  ifail = util_imin(0, -1);
  assert(ifail == -1);

  ifail = util_imin(1, 0);
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_math_square_modulus_int8
 *
 *****************************************************************************/

int test_util_math_square_modulus_int8(void) {

  int ifail = 0;

  {
    int8_t cv[3] = {1, 2, 3};
    int mod = util_square_modulus_int8(cv);
    assert(mod == 14);
    if (mod != 14) ifail = -1;
  }

  {
    int8_t cv[3] = {0, 0, 0};
    ifail = util_square_modulus_int8(cv);
    assert(ifail == 0);
  }

  return ifail;
}
