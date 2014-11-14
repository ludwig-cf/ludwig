/*****************************************************************************
 *
 *  unit_coords.c
 *
 *  Unit test for coords.c
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "coords.h"
#include "unit_control.h"

int do_test_coords_const(control_t * ctrl);
int do_test_coords_default(control_t * ctrl);
int do_test_coords_system(control_t * ctrl, int ntotal[3], int periodic[3]);
int do_test_coords_decomposition(control_t * ctrl, int decomp_ref[3]);

/*****************************************************************************
 *
 *  do_ut_coords
 *
 *****************************************************************************/

int do_ut_coords(control_t * ctrl) {

  assert(ctrl);
  do_test_coords_const(ctrl);
  do_test_coords_default(ctrl);
  /* do_test_coords_communicator(ctrl);*/

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_default
 *
 *****************************************************************************/

int do_test_coords_default(control_t * ctrl) {

  int ntotal_ref[3]   = {64, 64, 64};
  int periodic_ref[3] = {1, 1, 1};
  int decomposition_ref1[3] = {2, 2, 2};

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Default settings\n");

  try {
    coords_init();
    do_test_coords_system(ctrl, ntotal_ref, periodic_ref);
    do_test_coords_decomposition(ctrl, decomposition_ref1);
  }
  catch (MPITestFailedException) {
  }
  finally {
    coords_finish();
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_system
 *
 *****************************************************************************/

int do_test_coords_system(control_t * ctrl, int ntotal_ref[3],
			  int period_ref[3]) throws (MPITestFailedException) {

  int ntotal[3];

  assert(ctrl);
  control_verb(ctrl, "reference system %d %d %d\n",
	       ntotal_ref[X], ntotal_ref[Y], ntotal_ref[Z]);

  try {
    control_verb(ctrl, "ntotal\n");
    coords_ntotal(ntotal);
    control_macro_test(ctrl, ntotal[X] == ntotal_ref[X]);
    control_macro_test(ctrl, ntotal[Y] == ntotal_ref[Y]);
    control_macro_test(ctrl, ntotal[Z] == ntotal_ref[Z]);

    control_verb(ctrl, "default is_periodic()\n");
    control_macro_test(ctrl, is_periodic(X) == period_ref[X]);
    control_macro_test(ctrl, is_periodic(Y) == period_ref[Y]);
    control_macro_test(ctrl, is_periodic(Z) == period_ref[Z]);

    control_verb(ctrl, "default L()\n");
    control_macro_test_dbl_eq(ctrl, L(X), 64.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, L(Y), 64.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, L(Z), 64.0, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally { 
    if (control_allfail(ctrl)) throw(MPITestFailedException, "");
  }

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_decomposition
 *
 *  Slightly contrived: we can test the requested decomposition
 *  only if the total number of MPI tasks if correct to give
 *  that decomposition.
 *
 *****************************************************************************/

int do_test_coords_decomposition(control_t * ctrl, int decomp_ref[3])
  throws(MPITestFailedException) {

  int ntask;
  int ntotal[3];

  assert(ctrl);

  ntask = decomp_ref[X]*decomp_ref[Y]*decomp_ref[Z];

  coords_ntotal(ntotal);
  if (pe_size() != ntask) return 0;
  if (ntotal[X] % decomp_ref[X] != 0) return 0;
  if (ntotal[Y] % decomp_ref[Y] != 0) return 0;
  if (ntotal[Z] % decomp_ref[Z] != 0) return 0;

  /* Having met the above conditions, the test is valid... */
  control_verb(ctrl, "decomposition check %d %d %d\n",
	       decomp_ref[X], decomp_ref[Y], decomp_ref[Z]);

  try {
    control_macro_test(ctrl, cart_size(X) == decomp_ref[X]);
    control_macro_test(ctrl, cart_size(Y) == decomp_ref[Y]);
    control_macro_test(ctrl, cart_size(Z) == decomp_ref[Z]);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    if (control_allfail(ctrl)) throw(MPITestFailedException, "");
  }

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_const
 *
 *****************************************************************************/

int do_test_coords_const(control_t * ctrl) {

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "coords constants\n");

  try {
    control_verb(ctrl, "enum {X,Y,Z}\n");
    control_macro_test(ctrl, X == 0);
    control_macro_test(ctrl, Y == 1);
    control_macro_test(ctrl, Z == 2);

    control_verb(ctrl, "enum {XX, XY, ...}\n");
    control_macro_test(ctrl, XX == 0);
    control_macro_test(ctrl, XY == 1);
    control_macro_test(ctrl, XZ == 2);
    control_macro_test(ctrl, YY == 3);
    control_macro_test(ctrl, YZ == 4);
    control_macro_test(ctrl, ZZ == 5);

    control_verb(ctrl, "enum {FORW, BACK}\n");
    control_macro_test(ctrl, FORWARD  == 0);
    control_macro_test(ctrl, BACKWARD == 1);

    control_verb(ctrl, "Lmin\n");
    control_macro_test_dbl_eq(ctrl, Lmin(X), 0.5, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, Lmin(Y), 0.5, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, Lmin(Z), 0.5, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}
