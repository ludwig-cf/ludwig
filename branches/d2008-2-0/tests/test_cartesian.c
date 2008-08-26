/*****************************************************************************
 *
 *  t_cartesian.c
 *
 *  Cartesian Communicator.
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "cartesian.h"
#include "tests.h"

static int neighbour_rank(int, int, int);

int main(int argc, char ** argv) {

  int nlocal[3];
  int ntotal[3];
  int noffset[3];
  int dims[3]    = {1, 1, 1};
  int periods[3] = {1, 1, 1};
  int coords[3]  = {0, 0, 0};
  int rank       = 0;
  int n;

#ifdef _MPI_
  MPI_Comm communicator;
#endif

  pe_init(argc, argv);

  info("Running tests for the Cartesian Communicator...\n\n");

  coords_init();
  cart_init();

  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  get_N_local(nlocal);

#ifdef _MPI_
  info("Checking Cartesian communicator initialised...");
  communicator = cart_comm();
  MPI_Cart_get(communicator, 3, dims, periods, coords);
  info("yes\n");
#endif

  info("Checking cart_size(X) ...");
  test_assert(cart_size(X) == dims[X]);
  info("ok\n");

  info("Checking cart_size(Y) ...");
  test_assert(cart_size(Y) == dims[Y]);
  info("ok\n");

  info("Checking cart_size(Z) ...");
  test_assert(cart_size(Z) == dims[Z]);
  info("ok\n");

  info("Checking cart_coords(X) ...");
  test_assert(cart_coords(X) == coords[X]);
  info("ok\n");

  info("Checking cart_coords(Y) ...");
  test_assert(cart_coords(Y) == coords[Y]);
  info("ok\n");

  info("Checking cart_coords(Z) ...");
  test_assert(cart_coords(Z) == coords[Z]);
  info("ok\n");

  info("Checking periodity in X...");
  test_assert(is_periodic(X) == periods[X]);
  info("ok\n");

  info("Checking periodity in Y...");
  test_assert(is_periodic(Y) == periods[Y]);
  info("ok\n");

  info("Checking periodity in Z...");
  test_assert(is_periodic(Z) == periods[Z]);
  info("ok\n");

  info("Checking n_local[X] ...");
  test_assert(nlocal[X] == ntotal[X]/cart_size(X));
  info("ok\n");

  info("Checking n_local[Y] ...");
  test_assert(nlocal[Y] == ntotal[Y]/cart_size(Y));
  info("ok\n");

  info("Checking n_local[Z] ...");
  test_assert(nlocal[Z] == ntotal[Z]/cart_size(Z));
  info("ok\n");

  /* Check offsets */

  n = 0;
  get_N_offset(noffset);

  info("Checking n_offset(X)...");
  test_assert(noffset[X] == cart_coords(X)*nlocal[X]);
  info("ok\n");

  info("Checking n_offset(Y)...");
  test_assert(noffset[Y] == cart_coords(Y)*nlocal[Y]);
  info("ok\n");

  info("Checking n_offset(Z)...");
  test_assert(noffset[Z] == cart_coords(Z)*nlocal[Z]);
  info("ok\n");

#ifdef _MPI_
  MPI_Comm_rank(communicator, &rank);
#endif

  info("Checking Cartesian rank...");
  test_assert(cart_rank() == rank);
  info("ok\n");

  /* Check the neighbours */

  info("Checking FORWARD is 0...");
  test_assert(FORWARD == 0);
  info("yes\n");

  info("Checking BACKWARD is 1...");
  test_assert(BACKWARD == 1);
  info("yes\n");

  info("Checking FORWARD neighbours in X...");
  n = neighbour_rank(cart_coords(X)+1, cart_coords(Y), cart_coords(Z));
  test_assert(n == cart_neighb(FORWARD, X));
  info("ok\n");

  info("Checking BACKWARD neighbours in X...");
  n = neighbour_rank(cart_coords(X)-1, cart_coords(Y), cart_coords(Z));
  test_assert(n == cart_neighb(BACKWARD, X));
  info("ok\n");

  info("Checking FORWARD neighbours in Y...");
  n = neighbour_rank(cart_coords(X), cart_coords(Y)+1, cart_coords(Z));
  test_assert(n == cart_neighb(FORWARD, Y));
  info("ok\n");

  info("Checking BACKWARD neighbours in Y...");
  n = neighbour_rank(cart_coords(X), cart_coords(Y)-1, cart_coords(Z));
  test_assert(n == cart_neighb(BACKWARD, Y));
  info("ok\n");

  info("Checking FORWARD neighbours in Z...");
  n = neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)+1);
  test_assert(n == cart_neighb(FORWARD, Z));
  info("ok\n");

  info("Checking BACKWARD neighbours in Z...");
  n = neighbour_rank(cart_coords(X), cart_coords(Y), cart_coords(Z)-1);
  test_assert(n == cart_neighb(BACKWARD, Z));
  info("ok\n");


  /* This is for information only. */

#ifdef _MPI_

  info("\n");
  info("Overview\n");
  info("[rank] cartesian rank (X, Y, Z) cartesian order\n");

  for (n = 0; n < pe_size(); n++) {
    int index;
    MPI_Barrier(MPI_COMM_WORLD);

    /* This looks at whether cart_rank() is in the "natural" order */
    index = cart_size(Z)*cart_size(Y)*cart_coords(X) +
      cart_size(Z)*cart_coords(Y) + cart_coords(Z);

    if (n == pe_rank()) {
      printf("[%4d] %14d (%d, %d, %d) %d\n", pe_rank(), cart_rank(),
	     cart_coords(X), cart_coords(Y), cart_coords(Z), index);
    }
  }
  test_assert(1);

#endif

  pe_finalise();

  return 0;
}


int neighbour_rank(int nx, int ny, int nz) {

  int rank = 0;

#ifdef _MPI_
  int coords[3];
  coords[X] = nx;
  coords[Y] = ny;
  coords[Z] = nz;
  MPI_Cart_rank(cart_comm(), coords, &rank);
#endif

  return rank;
}
