/*****************************************************************************
 *
 *  test_coords.c
 *
 *  'Unit tests' for coords.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "tests.h"

static void test_coords_constants(void);
static void test_coords_system(const int ntotal[3], const int period[3]);
static void test_coords_decomposition(const int decomp_request[3]);
static void test_coords_communicator(void);
static void test_coords_cart_info(void);
static void test_coords_sub_communicator(void);
static int test_coords_periodic_comm(void);
static int neighbour_rank(int, int, int);

/*****************************************************************************
 *
 *  test_coords_suite
 *
 *****************************************************************************/

int test_coords_suite(void) {

  int ntotal_default[3] = {64, 64, 64};
  int periods_default[3] = {1, 1, 1};
  int decomposition_default[3] = {1, 1, 1};

  int ntotal_test1[3] = {1024, 1, 512};
  int periods_test1[3] = {1, 0, 1};
  int decomposition_test1[3] = {2, 1, 4};

  int ntotal_test2[3] = {1024, 1024, 1024};
  int periods_test2[3] = {1, 1, 1};
  int decomposition_test2[3] = {4, 4, 4};

  pe_init_quiet();

  /* info("Checking coords.c ...\n\n");*/

  test_coords_constants();

  /* Check the defaults, an the correct resetting of defaults. */

  /* info("\nCheck defaults...\n\n");*/
  test_coords_system(ntotal_default, periods_default);

  coords_init();
  test_coords_system(ntotal_default, periods_default);
  test_coords_decomposition(decomposition_default);
  test_coords_communicator();
  coords_finish();

  /* info("\nCheck reset of defaults...\n\n");*/
  test_coords_system(ntotal_default, periods_default);


  /* Now test 1 */

  coords_ntotal_set(ntotal_test1);
  coords_periodicity_set(periods_test1);
  coords_decomposition_set(decomposition_test1);

  coords_init();
  test_coords_system(ntotal_test1, periods_test1);
  test_coords_decomposition(decomposition_test1);
  test_coords_communicator();
  test_coords_cart_info();
  coords_finish();

  /* Now test 2 */

  coords_ntotal_set(ntotal_test2);
  coords_periodicity_set(periods_test2);
  coords_decomposition_set(decomposition_test2);

  coords_init();
  test_coords_system(ntotal_test2, periods_test2);
  test_coords_decomposition(decomposition_test2);
  test_coords_communicator();
  test_coords_cart_info();
  test_coords_sub_communicator();
  test_coords_periodic_comm();
  coords_finish();

  info("PASS     ./unit/test_coords\n");
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_constants
 *
 *  Test enums etc
 *
 *****************************************************************************/

void test_coords_constants(void) {

  /* info("Checking X Y Z enum... ");*/
  test_assert(X == 0);
  test_assert(Y == 1);
  test_assert(Z == 2);

  test_assert(XX == 0);
  test_assert(XY == 1);
  test_assert(XZ == 2);
  test_assert(YY == 3);
  test_assert(YZ == 4);
  /* info("ok\n");*/

  /* info("Checking FORWARD BACKWARD enum... ");*/
  test_assert(FORWARD == 0);
  test_assert(BACKWARD == 1);
  /* info("ok\n");*/

  /* info("Checking Lmin()... ");*/
  test_assert(fabs(Lmin(X) - 0.5) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(Lmin(Y) - 0.5) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(Lmin(Z) - 0.5) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  return;
}

/*****************************************************************************
 *
 *  test_coords_system
 *
 *  Check the results against the reference system
 *
 *****************************************************************************/

void test_coords_system(const int ntotal_ref[3], const int period_ref[3]) {

  /* info("Checking system N_total...");*/
  test_assert(N_total(X) == ntotal_ref[X]);
  test_assert(N_total(Y) == ntotal_ref[Y]);
  test_assert(N_total(Z) == ntotal_ref[Z]);
  /* info("yes\n");*/

  /* info("Checking periodicity ...");*/
  test_assert(is_periodic(X) == period_ref[X]);
  test_assert(is_periodic(Y) == period_ref[Y]);
  test_assert(is_periodic(Z) == period_ref[Z]);
  /* info("correct\n");*/

  /* info("Checking system L()...");*/
  test_assert(fabs(L(X) - 1.0*ntotal_ref[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(L(Y) - 1.0*ntotal_ref[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(L(Z) - 1.0*ntotal_ref[Z]) < TEST_DOUBLE_TOLERANCE);
  /* info("(ok)\n");*/

  return;
}

/*****************************************************************************
 *
 *  test_coods_decomposition
 *
 *  Check we got the requested decomposition, provided it is valid.
 *
 *****************************************************************************/

void test_coords_decomposition(const int decomp_request[3]) {

  int nproc;
  int ok = 1;

  nproc = decomp_request[X]*decomp_request[Y]*decomp_request[Z];
  if (pe_size() != nproc) ok = 0;
  if (N_total(X) % decomp_request[X] != 0) ok = 0;
  if (N_total(Y) % decomp_request[Y] != 0) ok = 0;
  if (N_total(Z) % decomp_request[Z] != 0) ok = 0;

  if (ok) {
    test_assert(cart_size(X) == decomp_request[X]);
    test_assert(cart_size(Y) == decomp_request[Y]);
    test_assert(cart_size(Z) == decomp_request[Z]);
  }

  return;
}

/*****************************************************************************
 *
 *  test_coords_communicator
 *
 *  Test the communicator stuff against reference values.
 *  We assume there has been a relevant call to test_coords_system()
 *  to check ntotal[] and periods[] are correct.
 *
 *****************************************************************************/

void test_coords_communicator(void) {

  MPI_Comm communicator;

  int nlocal[3];
  int ntotal[3];
  int noffset[3];
  int dims[3];
  int periods[3];
  int coords[3];
  int rank;
  int n;

  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  /* info("Checking Cartesian communicator initialised...");*/
  communicator = cart_comm();
  MPI_Cart_get(communicator, 3, dims, periods, coords);
  MPI_Comm_rank(communicator, &rank);
  /* info("yes\n");*/

  /* info("Checking Cartesian rank...");*/
  test_assert(cart_rank() == rank);
  /* info("ok\n");*/

  /* info("Checking cart_size() ...");*/
  test_assert(cart_size(X) == dims[X]);
  test_assert(cart_size(Y) == dims[Y]);
  test_assert(cart_size(Z) == dims[Z]);
  /* info("ok\n");*/

  /* info("Checking cart_coords() ...");*/
  test_assert(cart_coords(X) == coords[X]);
  test_assert(cart_coords(Y) == coords[Y]);
  test_assert(cart_coords(Z) == coords[Z]);
  /* info("ok\n");*/

  /* info("Checking periodity...");*/
  test_assert(is_periodic(X) == periods[X]);
  test_assert(is_periodic(Y) == periods[Y]);
  test_assert(is_periodic(Z) == periods[Z]);
  /* info("ok\n");*/

  /* info("Checking n_local[] ...");*/
  test_assert(nlocal[X] == ntotal[X]/cart_size(X));
  test_assert(nlocal[Y] == ntotal[Y]/cart_size(Y));
  test_assert(nlocal[Z] == ntotal[Z]/cart_size(Z));
  /* info("ok\n");*/

  /* info("Checking n_offset()...");*/
  test_assert(noffset[X] == cart_coords(X)*nlocal[X]);
  test_assert(noffset[Y] == cart_coords(Y)*nlocal[Y]);
  test_assert(noffset[Z] == cart_coords(Z)*nlocal[Z]);
  /* info("ok\n");*/

  /* Check the neighbours */

  /* info("Checking FORWARD neighbours in X...");*/
  n = neighbour_rank(cart_coords(X)+1, cart_coords(Y), cart_coords(Z));
  test_assert(n == cart_neighb(FORWARD, X));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in X...");*/
  n = neighbour_rank(cart_coords(X)-1, cart_coords(Y), cart_coords(Z));
  test_assert(n == cart_neighb(BACKWARD, X));
  /* info("ok\n");*/

  /* info("Checking FORWARD neighbours in Y...");*/
  n = neighbour_rank(cart_coords(X), cart_coords(Y)+1, cart_coords(Z));
  test_assert(n == cart_neighb(FORWARD, Y));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in Y...");*/
  n = neighbour_rank(cart_coords(X), cart_coords(Y)-1, cart_coords(Z));
  test_assert(n == cart_neighb(BACKWARD, Y));
  /* info("ok\n");*/

  /* info("Checking FORWARD neighbours in Z...");*/
  n = neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)+1);
  test_assert(n == cart_neighb(FORWARD, Z));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in Z...");*/
  n = neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)-1);
  test_assert(n == cart_neighb(BACKWARD, Z));
  /* info("ok\n");*/

  return;
}

/*****************************************************************************
 *
 *  test_coords_cart_info
 *
 *  Some information on the Cartesian communicator (strictly, not a test).
 *
 *****************************************************************************/

static void test_coords_cart_info(void) {

  int n, index;
  const int tag = 100;
  char string[FILENAME_MAX];

  MPI_Status status[1];

  /* info("\n");
  info("Overview\n");
  info("[rank] cartesian rank (X, Y, Z) cartesian order\n");*/

  /* This looks at whether cart_rank() is in the "natural" order */

  index = cart_size(Z)*cart_size(Y)*cart_coords(X) +
    cart_size(Z)*cart_coords(Y) + cart_coords(Z);
  /*
  sprintf(string, "[%4d] %14d (%d, %d, %d) %d\n", pe_rank(), cart_rank(),
	  cart_coords(X), cart_coords(Y), cart_coords(Z), index);
	
  info(string);
  */
  /* Pass everything to root to print in order. */

  if (pe_rank() != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < pe_size(); n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      /* info(string);*/
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_coords_sub_communicator
 *
 *  Look at some results of MPI_Cart_sub()
 *
 *****************************************************************************/

static void test_coords_sub_communicator(void) {

  int remainder[3];
  int n, rank1, rank2;
  const int tag = 100;
  char string[FILENAME_MAX];

  MPI_Comm sub_comm1;
  MPI_Comm sub_comm2;
  MPI_Status status[1];

  /* One-dimensional ub-communicator in Y */

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 0;

  MPI_Cart_sub(cart_comm(), remainder, &sub_comm1);
  MPI_Comm_rank(sub_comm1, &rank1);

  /* Two-dimensional sub-comminucator in YZ */

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 1;

  MPI_Cart_sub(cart_comm(), remainder, &sub_comm2);
  MPI_Comm_rank(sub_comm2, &rank2);

  /*
  info("\n");
  info("Sub-dimensional communicators\n");
  info("[rank] cartesian rank (X, Y, Z) -> Y 1-d YZ 2-d\n");

  sprintf(string, "[%4d] %14d (%d, %d, %d)        %d      %d\n",
	  pe_rank(), cart_rank(),
	  cart_coords(X), cart_coords(Y), cart_coords(Z), rank1, rank2);

  info(string);
  */
  /* Pass everything to root to print in order. */

  if (pe_rank() != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < pe_size(); n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      /* info(string);*/
    }
  }

  MPI_Comm_free(&sub_comm2);
  MPI_Comm_free(&sub_comm1);

  return;
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

int neighbour_rank(int nx, int ny, int nz) {

  int rank = 0;
  int coords[3];
  int periodic = 1;

  if (is_periodic(X) == 0 && (nx < 0 || nx >= cart_size(X))) periodic = 0;
  if (is_periodic(Y) == 0 && (ny < 0 || ny >= cart_size(Y))) periodic = 0;
  if (is_periodic(Z) == 0 && (nz < 0 || nz >= cart_size(Z))) periodic = 0;

  coords[X] = nx;
  coords[Y] = ny;
  coords[Z] = nz;

  rank = MPI_PROC_NULL;
  if (periodic) MPI_Cart_rank(cart_comm(), coords, &rank);

  /* Serial doesn't quite work out with the above */
  if (pe_size() == 1) rank = 0;

  return rank;
}

/*****************************************************************************
 *
 *  test_coords_periodic_comm
 *
 *****************************************************************************/

static int test_coords_periodic_comm(void) {

  int rank;
  int pforw, pback;
  int nsource, ndest;
  int coords[3];
  MPI_Comm pcomm;

  coords_periodic_comm(&pcomm);
  MPI_Comm_rank(pcomm, &rank);
  MPI_Cart_coords(pcomm, rank, 3, coords);

  coords_cart_shift(pcomm, X, FORWARD, &pforw);
  coords_cart_shift(pcomm, X, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, X, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  coords_cart_shift(pcomm, Y, FORWARD, &pforw);
  coords_cart_shift(pcomm, Y, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, Y, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  coords_cart_shift(pcomm, Z, FORWARD, &pforw);
  coords_cart_shift(pcomm, Z, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, Z, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  return 0;
}
