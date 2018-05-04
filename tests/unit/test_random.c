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

  /* info("Testing random number generators (may take a minute...)\n");
     info("Random sample size is %d\n\n", NLARGE);*/

  ran_init(pe);

  /* Check first number from serial generators */
  /*
  info("The first number from serial_uniform() is ...");
  r = ran_serial_uniform();
  info("%g ", r);
  test_assert(fabs(r - 0.0001946092) < TEST_FLOAT_TOLERANCE);
  info("(ok)\n");

  info("The next from serial_gaussian() is...");
  r = ran_serial_gaussian();
  info("%g ", r);
  test_assert(fabs(r - 1.6206822) < TEST_FLOAT_TOLERANCE);
  info("(ok)\n");
  */

  /* Check serial uniform statistics */

  /* info("Checking statistics for serial_uniform()...");*/

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_serial_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }
  /* info("\n");*/
  rtot = rtot/NLARGE;
  /*info("The mean is %g ", rtot);*/
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  /* info("(ok)\n");*/

  /* info("The minimum is %g ", rmin);*/
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  /* info("(ok)\n");*/

  /* info("The maximum is %g ", rmax);*/
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);
  /* info("(ok)\n");*/

  /* Check serial Gaussian statistics. Note that the variance is
   * computed by assuming the mean is indeed exactly zero. */

  /* info("Checking statistics for serial_gaussian()...");*/

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

  /* info("\n");*/
  rtot = rtot/NLARGE;
  /* info("The mean is %g ", rtot);*/
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);
  /* info("(ok)\n");*/

  /*info("The minimum is %g ", rmin);*/
  test_assert(rmin < -4.0);
  /*info("(ok)\n");*/

  /*info("The maximum is %g ", rmax);*/
  test_assert(rmax > 4.0);
  /*info("(ok)\n");*/

  rvar = rvar/NLARGE;
  /*info("The variance is %g ", rvar);*/
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  /* Check the parallel uniform generator. */

  /*info("Checking statistics for parallel_uniform()...");*/

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_parallel_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }
  /*info("\n");*/
  rtot = rtot/NLARGE;
  /*info("The mean is %g ", rtot);*/
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  /*info("The minimum is %g ", rmin);*/
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  /*info("The maximum is %g ", rmax);*/
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  /* Parallel Gaussian */

  /*info("Checking statistics for parallel_gaussian()...");*/

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

  /*info("\n");*/
  rtot = rtot/NLARGE;
  /*info("The mean is %g ", rtot);*/
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  /*info("The minimum is %g ", rmin);*/
  test_assert(rmin < -4.0);
  /*info("(ok)\n");*/

  /*info("The maximum is %g ", rmax);*/
  test_assert(rmax > 4.0);
  /*info("(ok)\n");*/

  rvar = rvar/NLARGE;
  /*info("The variance is %g ", rvar);*/
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);
  /*info("(ok)\n");*/

  pe_info(pe, "PASS     ./unit/test_random\n");
  pe_free(pe);

  return 0;
}
