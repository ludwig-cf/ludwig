/*****************************************************************************
 *
 *  t_coords.c
 *
 *  This tests coords.c (default and user input)
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "tests.h"


int main(int argc, char ** argv) {

  pe_init(argc, argv);

  info("Checking coords.c defaults...\n\n");

  coords_init();

  /* Check coordinate directions */

  info("Checking X enum... ");
  test_assert(X == 0);
  info("(ok)\n");

  info("Checking Y enum... ");
  test_assert(Y == 1);
  info("(ok)\n");

  info("Checking Z enum... ");
  test_assert(Z == 2);
  info("(ok)\n");

  /* Check default system size (lattice points) */

  info("Checking default system N_total[X] = 64...");
  test_assert(N_total(X) == 64);
  info("yes\n");

  info("Checking default system N_total[Y] = 64...");
  test_assert(N_total(Y) == 64);
  info("yes\n");

  info("Checking default system N_total[Z] = 64...");
  test_assert(N_total(Z) == 64);
  info("yes\n");

  /* Check default periodicity */

  info("Checking default periodicity in X...");
  test_assert(is_periodic(X));
  info("true\n");

  info("Checking default periodicity in Y...");
  test_assert(is_periodic(Y));
  info("true\n");

  info("Checking default periodicity in Z...");
  test_assert(is_periodic(Z));
  info("true\n");

  /* System length */

  info("Checking default system L(X)...");
  test_assert(fabs(L(X) - 64.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking default system L(Y)...");
  test_assert(fabs(L(Y) - 64.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking default system L(Z)...");
  test_assert(fabs(L(Z) - 64.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  /* System minimum Lmin() */

  info("Checking default Lmin(X)...");
  test_assert(fabs(Lmin(X) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking default Lmin(Y)...");
  test_assert(fabs(Lmin(Y) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking default Lmin(Z)...");
  test_assert(fabs(Lmin(Z) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");


  /* Now take some user input */

  info("\n");
  info("Checking user input test_coords_input1...\n");

  RUN_read_input_file("test_coords_input1");

  coords_init();

  /* Check user system size */

  info("Checking user system N_total[X] = 1024...");
  test_assert(N_total(X) == 1024);
  info("yes\n");

  info("Checking user system N_total[Y] = 1...");
  test_assert(N_total(Y) == 1);
  info("yes\n");

  info("Checking user system N_total[Z] = 512...");
  test_assert(N_total(Z) == 512);
  info("yes\n");

  /* Check user periodicity */

  info("Checking user periodicity in X is true...");
  test_assert(is_periodic(X));
  info("correct\n");

  info("Checking user periodicity in Y is false...");
  test_assert(is_periodic(Y) == 0);
  info("correct\n");

  info("Checking user periodicity in Z is true...");
  test_assert(is_periodic(Z));
  info("correct\n");

  /* System length */

  info("Checking user system L(X)...");
  test_assert(fabs(L(X) - 1024.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking user system L(Y)...");
  test_assert(fabs(L(Y) - 1.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking user system L(Z)...");
  test_assert(fabs(L(Z) - 512.0) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  /* System minimum Lmin() */

  info("Checking user Lmin(X)...");
  test_assert(fabs(Lmin(X) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking user Lmin(Y)...");
  test_assert(fabs(Lmin(Y) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  info("Checking user Lmin(Z)...");
  test_assert(fabs(Lmin(Z) - 0.5) < TEST_DOUBLE_TOLERANCE);
  info("(ok)\n");

  pe_finalise();

  return 0;
}
