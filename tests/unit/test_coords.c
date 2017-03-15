/*****************************************************************************
 *
 *  test_coords.c
 *
 *  'Unit tests' for coords.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "tests.h"

static int test_coords_constants(void);
static int test_coords_system(cs_t * cs, int ntotal[3], int period[3]);
static int test_coords_decomposition(cs_t * cs, int decomp_request[3]);
static int test_coords_communicator(cs_t * cs);
static int test_coords_cart_info(cs_t * cs);
static int test_coords_sub_communicator(cs_t * cs);
static int test_coords_periodic_comm(cs_t * cs);
static int neighbour_rank(cs_t * cs, int nx, int ny, int nz);

__host__ int do_test_coords_device1(pe_t * pe);
__global__ void do_test_coords_kernel1(cs_t * cs);


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
  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* info("Checking coords.c ...\n\n");*/
  
  test_coords_constants();

  /* Check the defaults, an the correct resetting of defaults. */

  /* info("\nCheck defaults...\n\n");*/
  cs_create(pe, &cs);
  test_coords_system(cs, ntotal_default, periods_default);

  cs_init(cs);
  test_coords_system(cs, ntotal_default, periods_default);
  test_coords_decomposition(cs, decomposition_default);
  test_coords_communicator(cs);
  cs_free(cs);

  /* Now test 1 */

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal_test1);
  cs_periodicity_set(cs, periods_test1);
  cs_decomposition_set(cs, decomposition_test1);

  cs_init(cs);
  test_coords_system(cs, ntotal_test1, periods_test1);
  test_coords_decomposition(cs, decomposition_test1);
  test_coords_communicator(cs);
  test_coords_cart_info(cs);
  cs_free(cs);

  /* Now test 2 */

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal_test2);
  cs_periodicity_set(cs, periods_test2);
  cs_decomposition_set(cs, decomposition_test2);

  cs_init(cs);
  test_coords_system(cs, ntotal_test2, periods_test2);
  test_coords_decomposition(cs, decomposition_test2);
  test_coords_communicator(cs);
  test_coords_cart_info(cs);
  test_coords_sub_communicator(cs);
  test_coords_periodic_comm(cs);
  cs_free(cs);

  /* Device tests */

  do_test_coords_device1(pe);

  pe_info(pe, "PASS     ./unit/test_coords\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_constants
 *
 *  Test enums etc
 *
 *****************************************************************************/

int test_coords_constants(void) {

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
  /*
  test_assert(fabs(Lmin(X) - 0.5) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(Lmin(Y) - 0.5) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(Lmin(Z) - 0.5) < TEST_DOUBLE_TOLERANCE);
  */
  /* info("ok\n");*/

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_system
 *
 *  Check the results against the reference system
 *
 *****************************************************************************/

int test_coords_system(cs_t * cs, int ntotal_ref[3], int period_ref[3]) {

  int ntotal[3];
  int periodic[3];
  double len[3];

  assert(cs);

  cs_ntotal(cs, ntotal);
  cs_periodic(cs, periodic);
  cs_ltot(cs, len);

  test_assert(ntotal[X] == ntotal_ref[X]);
  test_assert(ntotal[Y] == ntotal_ref[Y]);
  test_assert(ntotal[Z] == ntotal_ref[Z]);

  test_assert(periodic[X] == period_ref[X]);
  test_assert(periodic[Y] == period_ref[Y]);
  test_assert(periodic[Z] == period_ref[Z]);

  test_assert(fabs(len[X] - 1.0*ntotal_ref[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(len[Y] - 1.0*ntotal_ref[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(len[Z] - 1.0*ntotal_ref[Z]) < TEST_DOUBLE_TOLERANCE);

  return 0;
}

/*****************************************************************************
 *
 *  test_coods_decomposition
 *
 *  Check we got the requested decomposition, provided it is valid.
 *
 *****************************************************************************/

int test_coords_decomposition(cs_t * cs, int mpi_sz_req[3]) {

  int ok = 1;
  int ntask;
  int ntask_req;
  int ntotal[3];
  int mpisz[3];

  MPI_Comm comm;

  assert(cs);

  ntask_req = mpi_sz_req[X]*mpi_sz_req[Y]*mpi_sz_req[Z];

  cs_ntotal(cs, ntotal);
  cs_cartsz(cs, mpisz);
  cs_cart_comm(cs, &comm);

  MPI_Comm_size(comm, &ntask);

  if (ntask != ntask_req) ok = 0;
  if (ntotal[X] % mpi_sz_req[X] != 0) ok = 0;
  if (ntotal[Y] % mpi_sz_req[Y] != 0) ok = 0;
  if (ntotal[Z] % mpi_sz_req[Z] != 0) ok = 0;

  if (ok) {
    test_assert(mpisz[X] == mpi_sz_req[X]);
    test_assert(mpisz[Y] == mpi_sz_req[Y]);
    test_assert(mpisz[Z] == mpi_sz_req[Z]);
  }

  return 0;
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

int test_coords_communicator(cs_t * cs) {

  int nlocal[3];
  int ntotal[3];
  int noffset[3];
  int dims[3];
  int periods[3];
  int coords[3];
  int cart_coords[3];
  int mpisz[3];
  int periodic[3];
  int rank;
  int n;

  MPI_Comm comm;

  assert(cs);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  /* info("Checking Cartesian communicator initialised...");*/

  cs_cart_comm(cs, &comm);

  MPI_Cart_get(comm, 3, dims, periods, coords);
  MPI_Comm_rank(comm, &rank);
  /* info("yes\n");*/

  /* info("Checking Cartesian rank...");*/
  assert(cs_cart_rank(cs) == rank);
  /* info("ok\n");*/

  /* info("Checking cart_size() ...");*/
  cs_cartsz(cs, mpisz);
  assert(mpisz[X] == dims[X]);
  assert(mpisz[Y] == dims[Y]);
  assert(mpisz[Z] == dims[Z]);
  /* info("ok\n");*/

  /* info("Checking cart_coords() ...");*/
  cs_cart_coords(cs, cart_coords);
  assert(cart_coords[X] == coords[X]);
  assert(cart_coords[Y] == coords[Y]);
  assert(cart_coords[Z] == coords[Z]);
  /* info("ok\n");*/

  /* info("Checking periodity...");*/
  cs_periodic(cs, periodic);
  assert(periodic[X] == periods[X]);
  assert(periodic[Y] == periods[Y]);
  assert(periodic[Z] == periods[Z]);
  /* info("ok\n");*/

  /* info("Checking n_local[] ...");*/
  assert(nlocal[X] == ntotal[X]/mpisz[X]);
  assert(nlocal[Y] == ntotal[Y]/mpisz[Y]);
  assert(nlocal[Z] == ntotal[Z]/mpisz[Z]);
  /* info("ok\n");*/

  /* info("Checking n_offset()...");*/
  test_assert(noffset[X] == cart_coords[X]*nlocal[X]);
  test_assert(noffset[Y] == cart_coords[Y]*nlocal[Y]);
  test_assert(noffset[Z] == cart_coords[Z]*nlocal[Z]);
  /* info("ok\n");*/

  /* Check the neighbours */

  /* info("Checking FORWARD neighbours in X...");*/
  n = neighbour_rank(cs, cart_coords[X]+1, cart_coords[Y], cart_coords[Z]);
  assert(n == cs_cart_neighb(cs, CS_FORW, X));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in X...");*/
  n = neighbour_rank(cs, cart_coords[X]-1, cart_coords[Y], cart_coords[Z]);
  assert(n == cs_cart_neighb(cs, CS_BACK, X));
  /* info("ok\n");*/

  /* info("Checking FORWARD neighbours in Y...");*/
  n = neighbour_rank(cs, cart_coords[X], cart_coords[Y]+1, cart_coords[Z]);
  assert(n == cs_cart_neighb(cs, CS_FORW, Y));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in Y...");*/
  n = neighbour_rank(cs, cart_coords[X], cart_coords[Y]-1, cart_coords[Z]);
  assert(n == cs_cart_neighb(cs, CS_BACK, Y));
  /* info("ok\n");*/

  /* info("Checking FORWARD neighbours in Z...");*/
  n = neighbour_rank(cs, cart_coords[X], cart_coords[Y], cart_coords[Z]+1);
  assert(n == cs_cart_neighb(cs, CS_FORW, Z));
  /* info("ok\n");*/

  /* info("Checking BACKWARD neighbours in Z...");*/
  n = neighbour_rank(cs, cart_coords[X], cart_coords[Y], cart_coords[Z]-1);
  assert(n == cs_cart_neighb(cs, CS_BACK, Z));
  /* info("ok\n");*/

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_cart_info
 *
 *  Some information on the Cartesian communicator (strictly, not a test).
 *
 *****************************************************************************/

static int test_coords_cart_info(cs_t * cs) {

  int n;
  int rank, sz;
  const int tag = 100;
  char string[FILENAME_MAX];

  MPI_Status status[1];

  assert(cs);

  /* info("\n");
  info("Overview\n");
  info("[rank] cartesian rank (X, Y, Z) cartesian order\n");*/

  /* This looks at whether cart_rank() is in the "natural" order */
  /*
  index = cart_size(Z)*cart_size(Y)*cart_coords(X) +
    cart_size(Z)*cart_coords(Y) + cart_coords(Z);
  sprintf(string, "[%4d] %14d (%d, %d, %d) %d\n", pe_rank(), cart_rank(),
	  cart_coords(X), cart_coords(Y), cart_coords(Z), index);
	
  info(string);
  */
  /* Pass everything to root to print in order. */

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  if (rank != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < sz; n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      /* info(string);*/
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_sub_communicator
 *
 *  Look at some results of MPI_Cart_sub()
 *
 *****************************************************************************/

static int test_coords_sub_communicator(cs_t * cs) {

  int rank, sz;
  int remainder[3];
  int n, rank1, rank2;
  const int tag = 100;
  char string[FILENAME_MAX];

  MPI_Comm cartcomm;
  MPI_Comm sub_comm1;
  MPI_Comm sub_comm2;
  MPI_Status status[1];

  assert(cs);

  /* One-dimensional ub-communicator in Y */

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 0;

  cs_cart_comm(cs, &cartcomm);
  MPI_Cart_sub(cartcomm, remainder, &sub_comm1);
  MPI_Comm_rank(sub_comm1, &rank1);

  /* Two-dimensional sub-comminucator in YZ */

  remainder[X] = 0;
  remainder[Y] = 1;
  remainder[Z] = 1;

  MPI_Cart_sub(cartcomm, remainder, &sub_comm2);
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

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  if (rank != 0) {
    MPI_Ssend(string, FILENAME_MAX, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  else {
    for (n = 1; n < sz; n++) {
      MPI_Recv(string, FILENAME_MAX, MPI_CHAR, n, tag, MPI_COMM_WORLD, status);
      /* info(string);*/
    }
  }

  MPI_Comm_free(&sub_comm2);
  MPI_Comm_free(&sub_comm1);

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

int neighbour_rank(cs_t * cs, int nx, int ny, int nz) {

  int sz;
  int rank = 0;
  int coords[3];
  int periodic = 1;
  int is_periodic[3];
  int mpi_cartsz[3];
  MPI_Comm comm;

  assert(cs);

  cs_periodic(cs, is_periodic);
  cs_cartsz(cs, mpi_cartsz);
  cs_cart_comm(cs, &comm);

  if (is_periodic[X] == 0 && (nx < 0 || nx >= mpi_cartsz[X])) periodic = 0;
  if (is_periodic[Y] == 0 && (ny < 0 || ny >= mpi_cartsz[Y])) periodic = 0;
  if (is_periodic[Z] == 0 && (nz < 0 || nz >= mpi_cartsz[Z])) periodic = 0;

  coords[X] = nx;
  coords[Y] = ny;
  coords[Z] = nz;

  rank = MPI_PROC_NULL;
  if (periodic) MPI_Cart_rank(comm, coords, &rank);

  /* Serial doesn't quite work out with the above */
  MPI_Comm_size(comm, &sz);
  if (sz == 1) rank = 0; /* Fails in true MPI Comm_size = 1 */

  return rank;
}

/*****************************************************************************
 *
 *  test_coords_periodic_comm
 *
 *****************************************************************************/

static int test_coords_periodic_comm(cs_t * cs) {

  int rank;
  int pforw, pback;
  int nsource, ndest;
  int coords[3];
  MPI_Comm pcomm;

  assert(cs);
  cs_periodic_comm(cs, &pcomm);
  MPI_Comm_rank(pcomm, &rank);
  MPI_Cart_coords(pcomm, rank, 3, coords);

  cs_cart_shift(pcomm, X, FORWARD, &pforw);
  cs_cart_shift(pcomm, X, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, X, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  cs_cart_shift(pcomm, Y, FORWARD, &pforw);
  cs_cart_shift(pcomm, Y, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, Y, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  cs_cart_shift(pcomm, Z, FORWARD, &pforw);
  cs_cart_shift(pcomm, Z, BACKWARD, &pback);

  MPI_Cart_shift(pcomm, Z, 1, &nsource, &ndest);
  test_assert(pforw == ndest);
  test_assert(pback == nsource);

  return 0;
}

/******************************************************************************
 *
 *  do_test_coords_device1
 *
 *  Test default system size.
 *
 ******************************************************************************/

__host__ int do_test_coords_device1(pe_t * pe) {

  dim3 nblk, ntpb;
  cs_t * cstarget = NULL;
  cs_t * cs = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);
  cs_target(cs, &cstarget);

  kernel_launch_param(1, &nblk, &ntpb);
  ntpb.x = 1;

  tdpLaunchKernel(do_test_coords_kernel1, nblk, ntpb, 0, 0, cstarget);
  tdpDeviceSynchronize();

  cs_free(cs);

  return 0;
}

/******************************************************************************
 *
 *  do_test_coords_kernel1
 *
 ******************************************************************************/

__global__ void do_test_coords_kernel1(cs_t * cs) {

  int nhalo;
  int nsites;
  int mpisz[3];
  int mpicoords[3];
  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  double lmin[3];
  double ltot[3];

  assert(cs);

  cs_nhalo(cs, &nhalo);
  assert(nhalo == 1);

  cs_ntotal(cs, ntotal);
  assert(ntotal[X] == 64);
  assert(ntotal[Y] == 64);
  assert(ntotal[Z] == 64);

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);

  assert(nlocal[X] == ntotal[X]/mpisz[X]);
  assert(nlocal[Y] == ntotal[Y]/mpisz[Y]);
  assert(nlocal[Z] == ntotal[Z]/mpisz[Z]);

  cs_nsites(cs, &nsites);

  assert(nsites == (nlocal[X]+2*nhalo)*(nlocal[Y]+2*nhalo)*(nlocal[Z]+2*nhalo));

  cs_cart_coords(cs, mpicoords);
  cs_nlocal_offset(cs, noffset);

  assert(noffset[X] == mpicoords[X]*nlocal[X]);
  assert(noffset[Y] == mpicoords[Y]*nlocal[Y]);
  assert(noffset[Z] == mpicoords[Z]*nlocal[Z]);

  cs_lmin(cs, lmin);
  cs_ltot(cs, ltot);

  assert(fabs(lmin[X] - 0.5) < DBL_EPSILON);
  assert(fabs(lmin[Y] - 0.5) < DBL_EPSILON);
  assert(fabs(lmin[Z] - 0.5) < DBL_EPSILON);

  assert(fabs(ltot[X] - ntotal[X]) < DBL_EPSILON);
  assert(fabs(ltot[Y] - ntotal[Y]) < DBL_EPSILON);
  assert(fabs(ltot[Z] - ntotal[Z]) < DBL_EPSILON);

  return;
}
