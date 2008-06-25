/*****************************************************************************
 *
 *  test_halo.c
 *
 *  This is a more rigourous test of the halo swap code for the
 *  distributions than appears in test model.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2007)
 *
 *  $Id: test_halo.c,v 1.2 2008-06-25 17:17:48 erlend Exp $
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "tests.h"
#include "control.h"

static void test_halo_null(void);
static void test_halo(const int dim);
static void test_propagation();

int main(int argc, char ** argv) {

  pe_init(argc, argv);

  info("Checking distribution halo swaps...\n\n");

  RUN_read_input_file("input");
  init_control();
  coords_init();
  init_site();

  if(use_reduced_halos()) {
    info("Using reduced halos.\n");
  } else {
    info("Using full halos \n");
  }

  info("Test for null leakage...");
  test_halo_null();
  info("ok\n");

  info("Testing x-direction swap...");
  test_halo(X);
  info("ok\n");

  info("Testing y-direction swap...");
  test_halo(Y);
  info("ok\n");

  info("Testing z-direction swap...");
  test_halo(Z);
  info("ok\n");
  
  finish_site();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_halo_null
 *
 *  Null halo test. Make sure no halo information appears in the
 *  domain proper.
 *
 *****************************************************************************/

void test_halo_null() {

  int n_local[3], n[3];
  int index, p;
  double f_actual;

  get_N_local(n_local);

  /* Set entire distribution (all sites including halos) */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 1.0);
	}

      }
    }
  }

  /* Zero interior */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 0.0);
	}

      }
    }
  }

  /* Swap */

  halo_site();

  /* Check everywhere */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  f_actual = get_f_at_site(index, p);
	  test_assert(fabs(f_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_halo
 *
 *  Test the halo swap for the distributions for coordinate direction dim.
 *
 *****************************************************************************/

void test_halo(int dim) {

  int n_local[3], n[3];
  int offset[3];
  int index, p, d;

  double f_expect, f_actual;

  test_assert(dim == X || dim == Y || dim == Z);

  get_N_local(n_local);
  get_N_offset(offset);

  /* Zero entire distribution (all sites including halos) */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 0.0);
	}

      }
    }
  }

  /* Check neighbours in the given direction */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	if (n[X] == 1 || n[X] == n_local[X] ||
	    n[Y] == 1 || n[Y] == n_local[Y] ||
	    n[Z] == 1 || n[Z] == n_local[Z]) {

	  for (p = 0; p < NVEL; p++) {
	    set_f_at_site(index, p, offset[dim] + n[dim]);
	  }
	}

      }
    }
  }

  halo_site();

  /* Check the results (all sites).
   * The halo regions should contain a copy of the above, while the
   * interior sites are unchanged */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (d = 0; d < 3; d++) {

	  /* 'Left' side */
	  if (dim == d && n[d] == 0) {

	    f_expect = offset[dim];
	    if (cart_coords(dim) == 0) f_expect = L(dim);

	    for (p = 0; p < NVEL; p++) {
	      f_actual = get_f_at_site(index, p);
	      test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	    }
	  }

	  /* 'Right' side */
	  if (dim == d && n[d] == n_local[d] + 1) {

	    f_expect = offset[dim] + n_local[dim] + 1.0;
	    if (cart_coords(dim) == cart_size(dim) - 1) f_expect = 1.0;

	    for (p = 0; p < NVEL; p++) {
	      f_actual = get_f_at_site(index, p);
	      test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	    }
	  }

	}

      }
    }
  }

  return;
}

