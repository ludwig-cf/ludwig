/*****************************************************************************
 *
 *  test_site_map.c
 *
 *  Tests for site map interface.
 *
 *  $Id: test_site_map.c,v 1.1.2.1 2008-03-21 09:58:39 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "tests.h"

int main (int argc, char ** argv) {

  int ic, jc, kc;
  int nlocal[3];
  double v;

  pe_init(argc, argv);
  coords_init();

  get_N_local(nlocal);

  info("\nTesting site map interface\n");

  info("Site FLUID value is %d\n", FLUID);
  info("Site SOLID value is %d\n", SOLID);
  info("Site COLLOID value is %d\n", COLLOID);
  info("Site BOUNDARY value is %d\n", BOUNDARY);

  info("\nCalling site_map_init with default system size...");
  site_map_init();
  info("ok\n");

  /* All sites should be FLUID by default */

  info("Check all sites are fluid...");

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (jc = 0; jc <= nlocal[Y] + 1; jc++) {
      for (kc = 0; kc <= nlocal[Z] + 1; kc++) {
	test_assert(site_map_get_status(ic, jc, kc) == FLUID);
      }
    }
  }
  info("ok\n");

  /* Volume should be 64^3 */

  info("Checking fluid volume...");
  v = site_map_volume(FLUID);
  test_assert(fabs(v - L(X)*L(Y)*L(Z)) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");


  /* Check halo swap in each direction */

  info("Checking site_map_halo(X)...");

  for (jc = 1; jc <= nlocal[Y]; jc++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      site_map_set_status(1, jc, kc, SOLID);
      site_map_set_status(nlocal[X], jc, kc, BOUNDARY);
    }
  }

  site_map_halo();

  for (jc = 0; jc <= nlocal[Y] + 1; jc++) {
    for (kc = 0; kc <= nlocal[Z] + 1; kc++) {
      test_assert(site_map_get_status(0, jc, kc) == BOUNDARY);
      test_assert(site_map_get_status(nlocal[X]+1, jc, kc) == SOLID);
    }
  }

  info("ok\n");

  /* In parallel, the above formulation give multiple copies
   * of the solid/boundary sites at each subprocess boundary,
   * so compute volume accordingly. */

  info("Checking volume of solid sites...");

  v = site_map_volume(SOLID);
  test_assert(fabs(v - L(Y)*L(Z)*cart_size(X)) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  site_map_set_all(FLUID);

  info("Checking site halo(Y)...");

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      site_map_set_status(ic, 1, kc, SOLID);
      site_map_set_status(ic, nlocal[Y], kc, BOUNDARY);
    }
  }

  site_map_halo();

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (kc = 0; kc <= nlocal[Z] + 1; kc++) {
      test_assert(site_map_get_status(ic, 0, kc) == BOUNDARY);
      test_assert(site_map_get_status(ic, nlocal[Y]+1, kc) == SOLID);
    }
  }

  info("ok\n");

  info("Checking volume of boundary sites...");
  v = site_map_volume(BOUNDARY);
  test_assert(fabs(v - L(X)*L(Z)*cart_size(Y)) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  site_map_set_all(FLUID);

  info("Checking site halo(Z)...");

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      site_map_set_status(ic, jc, 1, SOLID);
      site_map_set_status(ic, jc, nlocal[Z], BOUNDARY);
    }
  }

  site_map_halo();

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (jc = 0; jc <= nlocal[Y] + 1; jc++) {
      test_assert(site_map_get_status(ic, jc, 0) == BOUNDARY);
      test_assert(site_map_get_status(ic, jc, nlocal[Z]+1) == SOLID);
    }
  }
  info("ok\n");

  info("Checking fluid volume...");
  v = site_map_volume(FLUID);
  test_assert(fabs(v - L(Y)*L(Z)*(L(X) - 2*cart_size(Z)))
	      < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  site_map_set_all(FLUID);

  info("Finsh site maps tests\n");
  site_map_finish();
  info("All site map tests ok!\n\n");

  pe_finalise();

  return 0;
}
