/*****************************************************************************
 *
 *  test_prop
 *
 *  Test propagation stage.
 *
 *  $Id: test_prop.c,v 1.4.2.5 2010-09-17 16:35:39 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 Ths University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "propagation.h"
#include "tests.h"

static void test_velocity(void);
static void test_source_destination(void);

int main(int argc, char ** argv) {

  pe_init(argc, argv);
  coords_init();
  distribution_init();

  info("Testing propagation...\n");
  info("Number of distributions is %d\n", distribution_ndist());

  info("\nFull halos...\n");

  distribution_halo_set_complete();
  test_velocity();
  test_source_destination();

  info("Full halo ok\n");

  info("Repeat with reduced halos...\n");

  distribution_halo_set_reduced();
  test_velocity();
  test_source_destination();

  info("Propagation ok\n");

  distribution_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_velocity
 *
 *  Check each distribution ends up with the same velocity index.
 *  This relies on the halo exchange working properly.
 *
 *****************************************************************************/

void test_velocity() {

  int n_local[3];
  int ic, jc, kc, index, p;
  int nd, ndist;
  double f_actual;

  coords_nlocal(n_local);
  ndist = distribution_ndist();

  /* Set test values */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    distribution_f_set(index, p, nd, 1.0*(p + nd));
	  }
	}

      }
    }
  }

  distribution_halo();
  propagation();

  /* Test */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);
	    test_assert(fabs(f_actual - 1.0*(p + nd)) < TEST_DOUBLE_TOLERANCE);
	  }
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_source_destination
 *
 *  Check each element of the distribution has propagated exactly one
 *  lattice spacing in the appropriate direction.
 *
 *  We use the global index as the test of the soruce.
 *  
 *****************************************************************************/

void test_source_destination() {

  int n_local[3], offset[3];
  int ic, jc, kc, index, p;
  int nd, ndist;
  int isource, jsource, ksource;
  double f_actual, f_expect;

  coords_nlocal(n_local);
  coords_nlocal_offset(offset);
  ndist = distribution_ndist();

  /* Set test values */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	f_actual = L(Y)*L(Z)*(offset[X] + ic) + L(Z)*(offset[Y] + jc) +
	  (offset[Z] + kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    distribution_f_set(index, p, nd, f_actual);
	  }
	}

      }
    }
  }

  distribution_halo();
  propagation();

  /* Test */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    isource = offset[X] + ic - cv[p][X];
	    if (isource == 0) isource += N_total(X);
	    if (isource == N_total(X) + 1) isource = 1;
	    jsource = offset[Y] + jc - cv[p][Y];
	    if (jsource == 0) jsource += N_total(Y);
	    if (jsource == N_total(Y) + 1) jsource = 1;
	    ksource = offset[Z] + kc - cv[p][Z];
	    if (ksource == 0) ksource += N_total(Z);
	    if (ksource == N_total(Z) + 1) ksource = 1;

	    f_expect = L(Y)*L(Z)*isource + L(Z)*jsource + ksource;
	    f_actual = distribution_f(index, p, nd);

	    /* In case of d2q9, propagation is only for kc = 1 */
	    if (NDIM == 2 && kc > 1) f_actual = f_expect;

	    test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	  }
	}

	/* Next site */
      }
    }
  }
  

  return;
}
