/*****************************************************************************
 *
 *  test_prop
 *
 *  Test propagation stage.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 Ths University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "propagation.h"
#include "tests.h"

static int do_test_velocity(lb_halo_enum_t halo);
static int do_test_source_destination(lb_halo_enum_t halo);

/*****************************************************************************
 *
 *  test_lb_prop_suite
 *
 *****************************************************************************/

int test_lb_prop_suite(void) {

  pe_init_quiet();
  coords_init();

  do_test_velocity(LB_HALO_FULL);
  do_test_velocity(LB_HALO_REDUCED);

  do_test_source_destination(LB_HALO_FULL);
  do_test_source_destination(LB_HALO_REDUCED);

  info("PASS     ./unit/test_prop\n");
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_velocity
 *
 *  Check each distribution ends up with the same velocity index.
 *  This relies on the halo exchange working properly.
 *
 *****************************************************************************/

int do_test_velocity(lb_halo_enum_t halo) {

  int nlocal[3];
  int ic, jc, kc, index, p;
  int nd;
  int nvel;
  int ndist = 2;
  double f_actual;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);

  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, halo);
  lb_nvel(lb, &nvel);

  coords_nlocal(nlocal);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f_set(lb, index, p, nd, 1.0*(p + nd));
	  }
	}

      }
    }
  }

  lb_halo(lb);
  lb_propagation(lb);

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f(lb, index, p, nd, &f_actual);
	    assert(fabs(f_actual - 1.0*(p + nd)) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_source_destination
 *
 *  Check each element of the distribution has propagated exactly one
 *  lattice spacing in the appropriate direction.
 *
 *  We use the global index as the test of the soruce.
 *  
 *****************************************************************************/

int do_test_source_destination(lb_halo_enum_t halo) {

  int nlocal[3], offset[3];
  int ic, jc, kc, index, p;
  int nd;
  int ndist = 2;
  int nvel;
  int isource, jsource, ksource;
  double f_actual, f_expect;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, halo);
  lb_nvel(lb, &nvel);

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	f_actual = L(Y)*L(Z)*(offset[X] + ic) + L(Z)*(offset[Y] + jc) +
	  (offset[Z] + kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
	    lb_f_set(lb, index, p, nd, f_actual);
	  }
	}

      }
    }
  }

  lb_halo(lb);
  lb_propagation(lb);

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < nvel; p++) {
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
	    lb_f(lb, index, p, nd, &f_actual);

	    /* In case of d2q9, propagation is only for kc = 1 */
	    if (NDIM == 2 && kc > 1) f_actual = f_expect;

	    assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}

	/* Next site */
      }
    }
  }

  lb_free(lb);

  return 0;
}
