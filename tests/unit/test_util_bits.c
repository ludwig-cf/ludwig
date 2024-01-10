/*****************************************************************************
 *
 *  test_util_bits.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "util_bits.h"

int test_util_bits_same(void);
int test_util_double_same(void);
int test_util_set_bit(void);
int test_util_unset_bit(void);

/*****************************************************************************
 *
 *  test_util_bits_suite
 *
 *****************************************************************************/

int test_util_bits_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_bits_same();
  test_util_double_same();
  test_util_set_bit();
  test_util_unset_bit();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_bits_same
 *
 *****************************************************************************/

int test_util_bits_same(void) {

  int ifail = 0;

  {
    int i1 = 1;
    int i2 = 1;
    ifail = util_bits_same(sizeof(int), &i1, &i2);
    assert(ifail == 1);
  }

  {
    int i1 = 2;
    int i2 = -2;
    ifail = util_bits_same(sizeof(int), &i1, &i2);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_double_same
 *
 *****************************************************************************/

int test_util_double_same(void) {

  int ifail = 0;

  {
    double a1 = 1.0;
    double a2 = a1;
    ifail = util_double_same(a1, a2);
    assert(ifail == 1);
  }

  {
    double a1 = +0.0;
    double a2 = -0.0;
    ifail = util_double_same(a1, a2);
    assert(ifail == 0);
  }

  {
    double a1 = 0.0;
    double a2 = 2.0;
    ifail = util_double_same(a1, a2);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_set_bit
 *
 *****************************************************************************/

int test_util_set_bit(void) {

  int ifail = 0;

  {
    int bit = 2;
    int mask = 0;
    mask = util_set_bit(mask, bit);
    if ((mask | bit) != bit) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_unset_bit
 *
 *****************************************************************************/

int test_util_unset_bit(void) {

  int ifail = 0;

  {
    int bit = 2;
    int mask = ~0;
    mask = util_unset_bit(mask,  bit);
    if ((mask ^ ~0) != bit) ifail = 0;
    assert(ifail == 0);
  }

  return ifail;
}
