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
int do_test_coords_nhalo(control_t * ctrl);
int do_test_coords_case1(control_t * ctrl);
int do_test_coords_case2(control_t * ctrl);
int do_test_coords_system(control_t * ctrl, int ntotal[3], int periodic[3]);
int do_test_coords_decomposition(control_t * ctrl, int decomp_ref[3]);
int do_test_coords_communicator(control_t * ctrl);
int do_test_coords_periodic_comm(control_t * ctrl);
int do_test_coords_cart_info(control_t * ctrl);
int do_test_coords_sub_comm_info(control_t * ctrl);
int ut_neighbour_rank(int nx, int ny, int nz, int * nrank);

/*****************************************************************************
 *
 *  do_ut_coords
 *
 *****************************************************************************/

int do_ut_coords(control_t * ctrl) {

  assert(ctrl);
  do_test_coords_const(ctrl);
  do_test_coords_default(ctrl);
  do_test_coords_nhalo(ctrl);
  do_test_coords_case1(ctrl);
  do_test_coords_case2(ctrl);
  do_test_coords_periodic_comm(ctrl);
  do_test_coords_cart_info(ctrl);
  do_test_coords_sub_comm_info(ctrl);

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
    do_test_coords_communicator(ctrl);
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
 *  do_test_coords_nhalo
 *
 *****************************************************************************/

int do_test_coords_nhalo(control_t * ctrl) {

  int nhalo_ref = 1;
  int nhalo;
  int nsites;
  int xs, ys, zs;
  int nlocal[3];

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Coords halo set to %d\n", nhalo_ref);

  coords_nhalo_set(nhalo_ref);
  coords_init();

  try {
    nhalo = coords_nhalo();
    control_macro_test(ctrl, nhalo == nhalo_ref);

    coords_strides(&xs, &ys, &zs);
    coords_nlocal(nlocal);
    control_macro_test(ctrl, zs == 1);
    control_macro_test(ctrl, ys == zs*(nlocal[Z] + 2*nhalo));
    control_macro_test(ctrl, xs == ys*(nlocal[Y] + 2*nhalo));
 
    nsites = coords_nsites();
    control_macro_test(ctrl, nsites = zs*ys*(nlocal[X] + 2*nhalo));
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    coords_finish();
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_case1
 *
 *****************************************************************************/

int do_test_coords_case1(control_t * ctrl) {

  int ntotal_ref1[3]   = {1024, 1, 512};
  int periodic_ref1[3] = {1, 0, 1};
  int decomposition_ref1[3] = {2, 1, 4};

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Case1: 2d\n");

  coords_ntotal_set(ntotal_ref1);
  coords_periodicity_set(periodic_ref1);
  coords_decomposition_set(decomposition_ref1);
  coords_init();

  try {
    do_test_coords_system(ctrl, ntotal_ref1, periodic_ref1);
    do_test_coords_decomposition(ctrl, decomposition_ref1);
    do_test_coords_communicator(ctrl);
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
 *  do_test_coords_case2
 *
 *****************************************************************************/

int do_test_coords_case2(control_t * ctrl) {

  int ntotal_ref1[3]   = {1024, 1024, 1024};
  int periodic_ref1[3] = {1, 1, 1};
  int decomposition_ref1[3] = {4, 4, 4};

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Case1: 3d\n");

  coords_ntotal_set(ntotal_ref1);
  coords_periodicity_set(periodic_ref1);
  coords_decomposition_set(decomposition_ref1);
  coords_init();

  try {
    do_test_coords_system(ctrl, ntotal_ref1, periodic_ref1);
    do_test_coords_decomposition(ctrl, decomposition_ref1);
    do_test_coords_communicator(ctrl);
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
    control_macro_test_dbl_eq(ctrl, L(X), 1.0*ntotal_ref[X], DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, L(Y), 1.0*ntotal_ref[Y], DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, L(Z), 1.0*ntotal_ref[Z], DBL_EPSILON);
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


/*****************************************************************************
 *
 *  test_coords_communicator
 *
 *****************************************************************************/

int do_test_coords_communicator(control_t * ctrl)
  throws (MPITestFailedException) {

  int nlocal[3];
  int ntotal[3];
  int noffset[3];
  int dims[3];
  int periods[3];
  int coords[3];
  int nr, rank;

  MPI_Comm comm;

  assert(ctrl);

  coords_ntotal(ntotal);
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  try {
    comm = cart_comm();

    control_verb(ctrl, "Cartesian communicator\n");
    control_macro_test(ctrl, comm != MPI_COMM_NULL);

    MPI_Cart_get(comm, 3, dims, periods, coords);
    MPI_Comm_rank(comm, &rank);

    control_verb(ctrl, "Checking Cartesian rank...\n");
    control_macro_test(ctrl, cart_rank() == rank);

    control_verb(ctrl, "Checking cart_size() ...\n");
    control_macro_test(ctrl, cart_size(X) == dims[X]);
    control_macro_test(ctrl, cart_size(Y) == dims[Y]);
    control_macro_test(ctrl, cart_size(Z) == dims[Z]);

    control_verb(ctrl, "Checking cart_coords() ...\n");
    control_macro_test(ctrl, cart_coords(X) == coords[X]);
    control_macro_test(ctrl, cart_coords(Y) == coords[Y]);
    control_macro_test(ctrl, cart_coords(Z) == coords[Z]);

    control_verb(ctrl, "Checking periodity...\n");
    control_macro_test(ctrl, is_periodic(X) == periods[X]);
    control_macro_test(ctrl, is_periodic(Y) == periods[Y]);
    control_macro_test(ctrl, is_periodic(Z) == periods[Z]);

    control_verb(ctrl, "Checking (uniform) nlocal[] ...\n");
    control_macro_test(ctrl, nlocal[X] == ntotal[X] / cart_size(X));
    control_macro_test(ctrl, nlocal[Y] == ntotal[Y] / cart_size(Y));
    control_macro_test(ctrl, nlocal[Z] == ntotal[Z] / cart_size(Z));

    control_verb(ctrl, "Checking (uniform) noffset()...\n");
    control_macro_test(ctrl, noffset[X] == cart_coords(X)*nlocal[X]);
    control_macro_test(ctrl, noffset[Y] == cart_coords(Y)*nlocal[Y]);
    control_macro_test(ctrl, noffset[Z] == cart_coords(Z)*nlocal[Z]);

    /* Check the neighbours */

    control_verb(ctrl, "Checking neighbour rank X+1...\n");
    ut_neighbour_rank(cart_coords(X)+1, cart_coords(Y), cart_coords(Z), &nr);
    control_macro_test(ctrl, nr == cart_neighb(FORWARD, X));

    control_verb(ctrl, "Checking neighbour rank X-1...\n");
    ut_neighbour_rank(cart_coords(X)-1, cart_coords(Y), cart_coords(Z), &nr);
    control_macro_test(ctrl, nr == cart_neighb(BACKWARD, X));

    control_verb(ctrl, "Checking neighbour rank Y+1...\n");
    ut_neighbour_rank(cart_coords(X), cart_coords(Y)+1, cart_coords(Z), &nr);
    control_macro_test(ctrl, nr == cart_neighb(FORWARD, Y));

    control_verb(ctrl, "Checking neighbour rank Y-1...\n");
    ut_neighbour_rank(cart_coords(X), cart_coords(Y)-1, cart_coords(Z), &nr);
    control_macro_test(ctrl, nr == cart_neighb(BACKWARD, Y));

    control_verb(ctrl, "Checking neighbour rank Z+1...\n");
    ut_neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)+1, &nr);
    control_macro_test(ctrl, nr == cart_neighb(FORWARD, Z));

    control_verb(ctrl, "Checking neighbour rank Z-1...\n");
    ut_neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)-1, &nr);
    control_macro_test(ctrl, nr == cart_neighb(BACKWARD, Z));
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
 *  neighbour_rank
 *
 *  Find the rank of the process at Cartesian coordinate (x,y,z)
 *  independently of the main code.
 *
 *  Calling MPI_Cart_rank() with out-of-range coordinates for
 *  non-periodic directions is erroneous, hence the extra logic.
 *
 *****************************************************************************/

int ut_neighbour_rank(int nx, int ny, int nz, int * nrank) {

  int coords[3];
  int periodic = 1;

  if (is_periodic(X) == 0 && (nx < 0 || nx >= cart_size(X))) periodic = 0;
  if (is_periodic(Y) == 0 && (ny < 0 || ny >= cart_size(Y))) periodic = 0;
  if (is_periodic(Z) == 0 && (nz < 0 || nz >= cart_size(Z))) periodic = 0;

  coords[X] = nx;
  coords[Y] = ny;
  coords[Z] = nz;

  *nrank = MPI_PROC_NULL;
  if (periodic) MPI_Cart_rank(cart_comm(), coords, nrank);

  /* Serial doesn't quite work out with the above */
  /*if (pe_size() == 1) *nrank = 0;*/ /* or stub library does not pass */

  return 0;
}
/*****************************************************************************
 *
 *  do_test_coords_cart_info
 *
 *  Just some information on the Cartesian communicator
 *
 *****************************************************************************/

int do_test_coords_cart_info(control_t * ctrl) {

  int n, index;
  int tag = 100;
  char string[FILENAME_MAX];

  MPI_Status status[1];

  assert(ctrl);

  coords_init();

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Information on Cartesian communicator\n"); 

  control_verb(ctrl, "Overview\n");
  control_verb(ctrl, "Decomposition %d %d %d\n",
	       cart_size(X), cart_size(Y), cart_size(Z));
  control_verb(ctrl, "[rank] cartesian rank (X, Y, Z) cartesian order\n");

  /* index is a reference "natural" order */

  index = cart_size(Z)*cart_size(Y)*cart_coords(X) +
    cart_size(Z)*cart_coords(Y) + cart_coords(Z);

  sprintf(string, "[%4d] %14d (%d, %d, %d) %d\n", pe_rank(), cart_rank(),
          cart_coords(X), cart_coords(Y), cart_coords(Z), index);

  control_verb(ctrl, string);

  /* Pass everything to root to print in order. */

  if (pe_rank() != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < pe_size(); n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      control_verb(ctrl, string);
    }
  }

  coords_finish();

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_sub_comm_info
 *
 *  Involving MPI_Cart_sub()
 *
 *****************************************************************************/

int do_test_coords_sub_comm_info(control_t * ctrl) {

  int remainder[3];
  int n, rank1, rank2;
  int tag = 100;
  char string[FILENAME_MAX];

  MPI_Comm comms1;
  MPI_Comm comms2;
  MPI_Status status[1];

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Sub-communicators\n");

  coords_init();

  /* One-dimensional ub-communicator in Y */

  remainder[X] = 0; remainder[Y] = 1; remainder[Z] = 0;

  MPI_Cart_sub(cart_comm(), remainder, &comms1);
  MPI_Comm_rank(comms1, &rank1);

  /* Two-dimensional sub-comminucator in YZ */

  remainder[X] = 0; remainder[Y] = 1; remainder[Z] = 1;

  MPI_Cart_sub(cart_comm(), remainder, &comms2);
  MPI_Comm_rank(comms2, &rank2);

  control_verb(ctrl, "Sub-dimensional communicators\n");
  control_verb(ctrl, "[rank] cartesian rank (X, Y, Z) -> Y 1-d YZ 2-d\n");

  sprintf(string, "[%4d] %14d (%d, %d, %d)        %d      %d\n",
          pe_rank(), cart_rank(),
          cart_coords(X), cart_coords(Y), cart_coords(Z), rank1, rank2);

  control_verb(ctrl , string);

  /* Pass everything to root to print in order. */

  if (pe_rank() != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < pe_size(); n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      control_verb(ctrl, string);
    }
  }

  MPI_Comm_free(&comms2);
  MPI_Comm_free(&comms1);
  coords_finish();

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_coords_periodic_comm
 *
 *****************************************************************************/

int do_test_coords_periodic_comm(control_t * ctrl) {

  int rank;
  int pforw, pback;
  int nsource, ndest;
  int coords[3];
  MPI_Comm pcomm;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Periodic communicator\n");

  coords_init();

  try {
    coords_periodic_comm(&pcomm);
    MPI_Comm_rank(pcomm, &rank);
    MPI_Cart_coords(pcomm, rank, 3, coords);

    coords_cart_shift(pcomm, X, FORWARD, &pforw);
    coords_cart_shift(pcomm, X, BACKWARD, &pback);
    MPI_Cart_shift(pcomm, X, 1, &nsource, &ndest);

    control_macro_test(ctrl, pforw == ndest);
    control_macro_test(ctrl, pback == nsource);

    coords_cart_shift(pcomm, Y, FORWARD, &pforw);
    coords_cart_shift(pcomm, Y, BACKWARD, &pback);
    MPI_Cart_shift(pcomm, Y, 1, &nsource, &ndest);

    control_macro_test(ctrl , pforw == ndest);
    control_macro_test(ctrl, pback == nsource);

    coords_cart_shift(pcomm, Z, FORWARD, &pforw);
    coords_cart_shift(pcomm, Z, BACKWARD, &pback);
    MPI_Cart_shift(pcomm, Z, 1, &nsource, &ndest);

    control_macro_test(ctrl, pforw == ndest);
    control_macro_test(ctrl, pback == nsource);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    coords_finish();
  }

  control_report(ctrl);

  return 0;
}
