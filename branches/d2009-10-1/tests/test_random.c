/*****************************************************************************
 *
 *  t_random.c
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
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "ran.h"
#include "tests.h"

#define NLARGE         10000000
#define STAT_TOLERANCE 0.001

int main(int argc, char ** argv) {

  double r;
  double rtot, rvar, rmin, rmax;
  double rhat[3], rmean[3];
  int    n;

  pe_init(argc, argv);

  info("Testing random number generators (may take a minute...)\n");
  info("Random sample size is %d\n\n", NLARGE);

  ran_init();

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
  /* Parallel generators */

#ifdef _MPI
  r = ran_parallel_uniform();
  verbose("PARALLEL %d ... %f\n", pe_rank(), r);
#endif

  /* Check serial uniform statistics */

  info("Checking statistics for serial_uniform()...");

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_serial_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }
  info("\n");
  rtot = rtot/NLARGE;
  info("The mean is %g ", rtot);
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The minimum is %g ", rmin);
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The maximum is %g ", rmax);
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);
  info("(ok)\n");

  /* Check serial Gaussian statistics. Note that the variance is
   * computed by assuming the mean is indeed exactly zero. */

  info("Checking statistics for serial_gaussian()...");

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

  info("\n");
  rtot = rtot/NLARGE;
  info("The mean is %g ", rtot);
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The minimum is %g ", rmin);
  test_assert(rmin < -4.0);
  info("(ok)\n");

  info("The maximum is %g ", rmax);
  test_assert(rmax > 4.0);
  info("(ok)\n");

  rvar = rvar/NLARGE;
  info("The variance is %g ", rvar);
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);
  info("(ok)\n");

  /* Check the parallel uniform generator. */

  info("Checking statistics for parallel_uniform()...");

  rtot = 0.0;
  rmin = 1.0;
  rmax = 0.0;

  for (n = 0; n < NLARGE; n++) {
    r = ran_parallel_uniform();
    rtot += r;
    if (r < rmin) rmin = r;
    if (r > rmax) rmax = r;
  }
  info("\n");
  rtot = rtot/NLARGE;
  info("The mean is %g ", rtot);
  test_assert(fabs(rtot - 0.5) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The minimum is %g ", rmin);
  test_assert(fabs(rmin - 0.0) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The maximum is %g ", rmax);
  test_assert(fabs(rmax - 1.0) < STAT_TOLERANCE);
  info("(ok)\n");

  /* Parallel Gaussian */

  info("Checking statistics for parallel_gaussian()...");

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

  info("\n");
  rtot = rtot/NLARGE;
  info("The mean is %g ", rtot);
  test_assert(fabs(rtot - 0.0) < STAT_TOLERANCE);
  info("(ok)\n");

  info("The minimum is %g ", rmin);
  test_assert(rmin < -4.0);
  info("(ok)\n");

  info("The maximum is %g ", rmax);
  test_assert(rmax > 4.0);
  info("(ok)\n");

  rvar = rvar/NLARGE;
  info("The variance is %g ", rvar);
  test_assert(fabs(rvar - 1.0) < STAT_TOLERANCE);
  info("(ok)\n");

  /* The serial unit vector routine */

  info("Testing serial_unit_vector()...\n");

  rmin = 0.0;
  rmax = 0.0;
  rmean[0] = 0.0;
  rmean[1] = 0.0;
  rmean[2] = 0.0;

  for (n = 0; n < NLARGE; n++) {
    ran_serial_unit_vector(rhat);
    rvar = rhat[0]*rhat[0] + rhat[1]*rhat[1] + rhat[2]*rhat[2];
    test_assert(fabs(rvar - 1.0) < TEST_DOUBLE_TOLERANCE);
    rmean[0] += rhat[0];
    rmean[1] += rhat[1];
    rmean[2] += rhat[2];
    rmin = dmin(rmin, rhat[0]);
    rmin = dmin(rmin, rhat[1]);
    rmin = dmin(rmin, rhat[2]);
    rmax = dmax(rmax, rhat[0]);
    rmax = dmax(rmax, rhat[1]);
    rmax = dmax(rmax, rhat[2]);
  }

  info("Unit vector modulus is %f (ok)\n", rvar);

  test_assert(rmin >= -1.0);
  info("Component min is %g (ok)\n", rmin);
  
  test_assert(rmax <= +1.0);
  info("Component max is %g (ok)\n", rmax);

  rmean[0] /= NLARGE;
  rmean[1] /= NLARGE;
  rmean[2] /= NLARGE;

  test_assert(fabs(rmean[0]) < STAT_TOLERANCE);
  info("Component <X> is %g (ok)\n", rmean[0]);
  test_assert(fabs(rmean[1]) < STAT_TOLERANCE);
  info("Component <Y> is %g (ok)\n", rmean[1]);
  test_assert(fabs(rmean[2]) < STAT_TOLERANCE);
  info("Component <Z> is %g (ok)\n", rmean[2]);

  /* Repeat for the parallel version. */

  info("Testing parallel_unit_vector()...\n");

  rmin = 0.0;
  rmax = 0.0;
  rmean[0] = 0.0;
  rmean[1] = 0.0;
  rmean[2] = 0.0;

  for (n = 0; n < NLARGE; n++) {
    ran_parallel_unit_vector(rhat);
    rvar = rhat[0]*rhat[0] + rhat[1]*rhat[1] + rhat[2]*rhat[2];
    test_assert(fabs(rvar - 1.0) < TEST_DOUBLE_TOLERANCE);
    rmean[0] += rhat[0];
    rmean[1] += rhat[1];
    rmean[2] += rhat[2];
    rmin = dmin(rmin, rhat[0]);
    rmin = dmin(rmin, rhat[1]);
    rmin = dmin(rmin, rhat[2]);
    rmax = dmax(rmax, rhat[0]);
    rmax = dmax(rmax, rhat[1]);
    rmax = dmax(rmax, rhat[2]);
  }

  info("Unit vector modulus is %f (ok)\n", rvar);

  test_assert(rmin >= -1.0);
  info("Component min is %g (ok)\n", rmin);
  
  test_assert(rmax <= +1.0);
  info("Component max is %g (ok)\n", rmax);

  rmean[0] /= NLARGE;
  rmean[1] /= NLARGE;
  rmean[2] /= NLARGE;

  test_assert(fabs(rmean[0]) < STAT_TOLERANCE);
  info("Component <X> is %g (ok)\n", rmean[0]);
  test_assert(fabs(rmean[1]) < STAT_TOLERANCE);
  info("Component <Y> is %g (ok)\n", rmean[1]);
  test_assert(fabs(rmean[2]) < STAT_TOLERANCE);
  info("Component <Z> is %g (ok)\n", rmean[2]);


  info("\nRandom number tests ok\n\n");

  pe_finalise();

  return 0;
}
