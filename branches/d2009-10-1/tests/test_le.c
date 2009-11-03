/*****************************************************************************
 *
 *  test_leesedwards.c
 *
 *
 *  $Id: test_le.c,v 1.1 2009-08-07 16:32:52 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "tests.h"

static void test_parallel1(void);

int main (int argc, char ** argv) {

  pe_init(argc, argv);
  coords_init();

  test_parallel1();

  info("\nLees Edwards tests completed ok.\n");
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_parallel1
 *
 *  Check the parallel transformation stuff makes sense.
 *
 *****************************************************************************/

void test_parallel1(void) {

  const int nplane = 2;
  int nplane_local;
  int n;
  int jdy;
  int jlocal, j1, j1mod;
  int n1, n2;
  int py;
  int precv_rank_left, precv_rank_right;
  int nrank_r[2], nrank_s[2];

  int nlocal[3];
  int noffset[3];

  const double uy_set = 0.25;
  double uy;
  double fr;

  double dy;

  MPI_Comm comm;

  le_set_nplane_total(nplane);
  le_set_plane_uymax(uy_set);
  le_init();

  info("\nLees Edwards test (constant speed)...\n");
  info("Total number of planes in set correctly... ");
  test_assert(le_get_nplane_total() == nplane);
  info("yes\n");

  info("Local number of planes set correctly... ");
  nplane_local = nplane / cart_size(Y);
  test_assert(le_get_nplane_local() == nplane_local);
  info("yes\n");

  info("Plane maximum velocity set correctly... ");
  uy = le_plane_uy_max();
  test_assert(fabs(uy - uy_set) < TEST_DOUBLE_TOLERANCE);
  info("yes\n");


  /* Check displacement calculations. Run to a displacement which is
   * at least a couple of periodic images. */

  get_N_local(nlocal);
  get_N_offset(noffset);

  comm = le_communicator();

  for (n = -5000; n <= 5000; n++) {

    /* Set the displacement dy, and take modulo L(Y), which should
     * give -L(Y) < dy < +L(Y) */

    dy = uy_set*n;
    dy = fmod(dy, L(Y));

    test_assert(dy > -L(Y));
    test_assert(dy < +L(Y));

    /* The integral part of the displacement jdy and the fractional
     * part are... */

    jdy = floor(dy);
    fr = dy - jdy;

    test_assert(jdy < N_total(Y));
    test_assert(jdy >= - N_total(Y));
    test_assert(fr >= 0.0);
    test_assert(fr <= 1.0);

    /* The first point in global coordinates involved in the buffer
     * exchange is j1, displaced -(jdy+1) from the local leftmost
     * point. Modular arithmetic ensures 1 <= j1 <= N_y.  */

    jlocal = noffset[Y] + 1;
    j1 = 1 + (jlocal - jdy - 2 + 2*N_total(Y)) % N_total(Y);

    test_assert(j1 >= 1);
    test_assert(j1 <= N_total(Y));

    /* The corresponding local coordinate is j1mod */

    j1mod = 1 + j1 % nlocal[Y];

    test_assert(j1mod >= 1);
    test_assert(j1mod <= nlocal[Y]);

    /* Number of points we can grab from the left processor is n1 */

    n1 = nlocal[Y] - j1mod + 1;

    test_assert(n1 >= 1);
    test_assert(n1 <= nlocal[Y]);

    /* Number of points to grab from right procesor is n2 which is
     * (nlocal[Y] + 1) - n1 */

    n2 = j1mod;

    test_assert(n2 >= 1);
    test_assert(n2 <= nlocal[Y]);
    test_assert((n1 + n2) == (nlocal[Y] + 1));

    py = (j1 - 1) / nlocal[Y];
    test_assert(py >= 0);
    test_assert(py < cart_size(Y));
    MPI_Cart_rank(comm, &py, &precv_rank_left);
    py = 1 + (j1 - 1) / nlocal[Y];
    test_assert(py >= 1);
    test_assert(py <= cart_size(Y));
    MPI_Cart_rank(comm, &py, &precv_rank_right);
  }

  le_finish();

  return;
}
