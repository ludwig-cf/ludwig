/*****************************************************************************
 *
 *  test_prop
 *
 *  Test propagation stage (single distribution).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) Ths University of Edinburgh (2007)
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
  init_site();

  info("Testing propagation...\n");

  test_velocity();
  test_source_destination();

  info("Propagation ok\n");

  finish_site();
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
  double f_actual;

  get_N_local(n_local);

  /* Set test values */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = index_site(ic, jc, kc);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, (double) p);
	}

      }
    }
  }

  halo_site();
  propagation();

  /* Test */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = index_site(ic, jc, kc);

	for (p = 0; p < NVEL; p++) {
	  f_actual = get_f_at_site(index, p);
	  test_assert(fabs(f_actual - p) < TEST_DOUBLE_TOLERANCE);
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
  int isource, jsource, ksource;
  double f_actual, f_expect;

  get_N_local(n_local);
  get_N_offset(offset);

  /* Set test values */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = index_site(ic, jc, kc);

	f_actual = L(Y)*L(Z)*(offset[X] + ic) + L(Z)*(offset[Y] + jc) +
	  (offset[Z] + kc);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, f_actual);
	}

      }
    }
  }

  halo_site();
  propagation();

  /* Test */

  for (ic = 1; ic <= n_local[X]; ic++) {
    for (jc = 1; jc <= n_local[Y]; jc++) {
      for (kc = 1; kc <= n_local[Z]; kc++) {

	index = index_site(ic, jc, kc);

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
	  f_actual = get_f_at_site(index, p);

	  test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	}

      }
    }
  }
  

  return;
}
