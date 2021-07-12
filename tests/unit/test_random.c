/*****************************************************************************
 *
 *  test_random.c
 *
 *  Random number generator tests. Note that:
 *
 *   - the test always takes the default seed
 *   - statistics are based on a sample of NLARGE numbers
 *
 *  For a sample of 10 million, a tolerance of 0.001 will pass
 *  the statistical tests. Larger samples might have stricter
 *  tolerance.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "ran.h"
#include "util.h"
#include "tests.h"

#define NLARGE         10000000
#define STAT_TOLERANCE 0.001

/*****************************************************************************
 *
 *  test_random_suite
 *
 *****************************************************************************/

int test_random_suite(void) {

  int n;
  double r;
  double rtot, rvar, rmin, rmax;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* "Testing random number generators (may take a minute...)"
     "Random sample size is %d\n\n", NLARGE */

  ran_init(pe);

  /* Check first,second  number from serial generators */
  /* This may be historical:
   * fabs(r - 0.0001946092) < TEST_FLOAT_TOLERANCE
   * fabs(r - 1.6206822)    < TEST_FLOAT_TOLERANCE
   */

  /* Check serial uniform statistics */

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_serial_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }

  rtot = rtot/NLARGE;
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);

  /* Check serial Gaussian statistics. Note that the variance is
   * computed by assuming the mean is indeed exactly zero. */

  rtot = 0.0;
  rvar = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_serial_gaussian();
    rtot += r;
    rvar += (r*r);
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }

  rtot = rtot/NLARGE;
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);

  test_assert(rmin < -4.0);
  test_assert(rmax > 4.0);

  rvar = rvar/NLARGE;
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);

  /* Check the parallel uniform generator. */

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_parallel_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }

  rtot = rtot/NLARGE;
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);

  /* Parallel Gaussian */

  rtot = 0.0;
  rvar = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_parallel_gaussian();
    rtot += r;
    rvar += (r*r);
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }

  rtot = rtot/NLARGE;
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);
  test_assert(rmin < -4.0);
  test_assert(rmax > 4.0);

  rvar = rvar/NLARGE;
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);

  pe_info(pe, "PASS     ./unit/test_random\n");
  pe_free(pe);

  return 0;
}
