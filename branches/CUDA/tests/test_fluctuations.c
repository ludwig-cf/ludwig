/*****************************************************************************
 *
 *  test_fluctuations.c
 *
 *  Test the basic fluctuation generator type.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "fluctuations.h"
#include "tests.h"

/*****************************************************************************
 *
 *  main.c
 *
 *  This is a unit test which checks the interface does what it should
 *  do, nothing more. Specifically, there are no statistical tests.
 * 
 *****************************************************************************/

int main (int argc, char ** argv) {

  fluctuations_t * f;

  double a1, a2;
  double r[NFLUCTUATION];
  unsigned int state_ref[NFLUCTUATION_STATE] = {123, 456, 78, 9};
  unsigned int state[NFLUCTUATION_STATE];

  a1 = sqrt(2.0 + sqrt(2.0));
  a2 = sqrt(2.0 - sqrt(2.0));

  pe_init(argc, argv);
  f = fluctuations_create(1);

  fluctuations_reap(f, 0, r);
  fluctuations_state(f, 0, state);

  /* The initial state should be zero, and one call to 'reap' should
   * move us to... */ 

  test_assert(state[0] == 1234567);
  test_assert(state[1] == 0);
  test_assert(state[2] == 0);
  test_assert(state[3] == 0);

  /* Check the values */

  test_assert(fabs(r[0] - -a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[1] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[2] - +a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[3] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[4] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[5] - -a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[6] - -a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[7] - -a1) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[8] - -a1) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[9] - -a1) < TEST_DOUBLE_TOLERANCE);

  /* Set some state and make sure is is correct */

  fluctuations_state_set(f, 0, state_ref);
  fluctuations_state(f, 0, state);

  test_assert(state[0] == state_ref[0]);
  test_assert(state[1] == state_ref[1]);
  test_assert(state[2] == state_ref[2]);
  test_assert(state[3] == state_ref[3]);

  fluctuations_reap(f, 0, r);

  test_assert(fabs(r[0] - +a1) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[1] - -a1) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[2] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[3] - +a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[4] - +a2) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[5] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[6] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[7] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[8] - 0.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r[9] - 0.0) < TEST_DOUBLE_TOLERANCE);

  fluctuations_destroy(f);
  pe_finalise();

  return 0;
}
