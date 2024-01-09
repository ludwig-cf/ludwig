/*****************************************************************************
 *
 *  test_cs_limits.c
 *
 *  Some basic sanity checks.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "cs_limits.h"

int test_cs_limits(int imin, int imax, int jmin, int jmax, int kmin, int kmax);
int test_cs_limits_size(cs_limits_t lim);
int test_cs_limits_ic(cs_limits_t lim);
int test_cs_limits_jc(cs_limits_t lim);
int test_cs_limits_kc(cs_limits_t lim);
int test_cs_limits_index(cs_limits_t lim);

/*****************************************************************************
 *
 *  test_cs_limits_suite
 *
 *****************************************************************************/

int test_cs_limits_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* A selection of (imin, imax, jmin, jmax, kmin, kmax) */
  test_cs_limits( 1, 16, 1,  1, 1,  1);
  test_cs_limits( 1, 16, 1,  8, 1,  4);
  test_cs_limits(-1, 18, 0,  9, 1,  1);
  test_cs_limits( 1,  1, 0,  0, 1, 16);
  test_cs_limits(-3, -1, 1,  1, 2, 16);

  pe_info(pe, "PASS     ./unit/test_cs_limits\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_cs_limits
 *
 *****************************************************************************/

int test_cs_limits(int imin, int imax, int jmin, int jmax, int kmin, int kmax) {

  cs_limits_t lim = {imin, imax, jmin, jmax, kmin, kmax};

  test_cs_limits_size(lim);
  test_cs_limits_kc(lim);
  test_cs_limits_jc(lim);
  test_cs_limits_ic(lim);
  test_cs_limits_index(lim);

  return 0;
}
/*****************************************************************************
 *
 *  test_cs_limits_size
 *
 *****************************************************************************/

int test_cs_limits_size(cs_limits_t lim) {

  int ifail = 0;

  int nx = 1 + lim.imax - lim.imin;
  int ny = 1 + lim.jmax - lim.jmin;
  int nz = 1 + lim.kmax - lim.kmin;

  assert(nx*ny*nz == cs_limits_size(lim));
  if (nx*ny*nz != cs_limits_size(lim)) ifail += 1;

  return ifail;
}

/*****************************************************************************
 *
 *  test_cs_limits_ic
 *
 *****************************************************************************/

int test_cs_limits_ic(cs_limits_t lim) {

  int ifail = 0;

  {
    /* Flat index zero must be imin. */
    int ic = cs_limits_ic(lim, 0);
    assert(ic == lim.imin);
    if (ic != lim.imin) ifail += 1;
  }

  {
    /* Modular arithmetic check ... topmost flat index must be imax ... */

    int ic = cs_limits_ic(lim, cs_limits_size(lim) - 1);
    assert(ic == lim.imax);
    if (ic != lim.imax) ifail += 1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_cs_limits_jc
 *
 *****************************************************************************/

int test_cs_limits_jc(cs_limits_t lim) {

  int ifail = 0;

  {
    /* Flat index 0 must be jmin. */
    int jc = cs_limits_jc(lim, 0);
    assert(jc == lim.jmin);
    if (jc != lim.jmin) ifail += 1;
  }

  {
    /* Modular arithmetic  ...*/
    int jc = cs_limits_jc(lim, cs_limits_size(lim) - 1);
    assert(jc == lim.jmax);
    if (jc != lim.jmax) ifail += 1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_cs_limits_kc
 *
 *****************************************************************************/

int test_cs_limits_kc(cs_limits_t lim) {

  int ifail = 0;

  {
    /* Flat index 0 must be kmin */
    int kc = cs_limits_kc(lim, 0);
    assert(kc == lim.kmin);
    if (kc != lim.kmin) ifail += 1;
  }

  {
    /* Modular arithemtic check ...  */
    int kc = cs_limits_kc(lim, cs_limits_size(lim) - 1);
    assert(kc == lim.kmax);
    if (kc != lim.kmax) ifail += 1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_cs_limits_index
 *
 *****************************************************************************/

int test_cs_limits_index(cs_limits_t lim) {

  int ifail = 0;

  {
    /* Flat index 0 is (imin, jmin, kmin) */
    int iflat = cs_limits_index(lim, lim.imin, lim.jmin, lim.kmin);
    assert(iflat == 0);
    if (iflat != 0) ifail += 1;
  }

  {
    /* Flat index for (imax, jmax, kmax) ... */
    int iflat = cs_limits_index(lim, lim.imax, lim.jmax, lim.kmax);
    assert(iflat == cs_limits_size(lim) - 1);
    if (iflat != cs_limits_size(lim) - 1) ifail += 1;
  }

  return ifail;
}
