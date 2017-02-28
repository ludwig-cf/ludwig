/*****************************************************************************
 *
 *  test_leesedwards.c
 *
 *  Lees Edwards structure
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "physics.h"
#include "util.h"
#include "tests.h"

static int test_parallel1(pe_t * pe, cs_t * cs);
static int test_le_parallel2(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_le_suite
 *
 *****************************************************************************/

int test_le_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  physics_t * phys = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  cs_create(pe, &cs);
  cs_init(cs);

  physics_create(pe, &phys);

  test_parallel1(pe, cs);
  test_le_parallel2(pe, cs);

  physics_free(phys);
  cs_free(cs);

  pe_info(pe, "PASS     ./unit/test_le\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_parallel1
 *
 *  Check the parallel transformation stuff makes sense.
 *
 *****************************************************************************/

int test_parallel1(pe_t * pe, cs_t * cs) {

  const int nplane = 2;
  int nplane_local;
  int n;
  int jdy;
  int jlocal, j1, j1mod;
  int n1, n2;
  int py;
  int precv_rank_left, precv_rank_right;

  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  int cartsz[3];

  const double uy_set = 0.25;
  double uy;
  double fr;

  double dy;
  double len[3];

  lees_edw_info_t myinfo = {0};
  lees_edw_info_t * info = &myinfo;
  lees_edw_t * le = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);

  info->nplanes = nplane;
  info->uy = uy_set;

  cs_cartsz(cs, cartsz);
  lees_edw_create(pe, cs, info, &le);

  /*info("\nLees Edwards test (constant speed)...\n");
    info("Total number of planes in set correctly... ");*/
  test_assert(lees_edw_nplane_total(le) == nplane);
  /*info("yes\n");*/

  /* info("Local number of planes set correctly... ");*/
  nplane_local = nplane / cartsz[X];
  test_assert(lees_edw_nplane_local(le) == nplane_local);
  /* info("yes\n");*/

  /* info("Plane maximum velocity set correctly... ");*/
  lees_edw_plane_uy(le, &uy);
  test_assert(fabs(uy - uy_set) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/


  /* Check displacement calculations. Run to a displacement which is
   * at least a couple of periodic images. */

  cs_ltot(cs, len);
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  lees_edw_comm(le, &comm);

  for (n = -5000; n <= 5000; n++) {

    /* Set the displacement dy, and take modulo L(Y), which should
     * give -L(Y) < dy < +L(Y) */

    dy = uy_set*n;
    dy = fmod(dy, len[Y]);

    test_assert(dy > -len[Y]);
    test_assert(dy < +len[Y]);

    /* The integral part of the displacement jdy and the fractional
     * part are... */

    jdy = floor(dy);
    fr = dy - jdy;

    test_assert(jdy < ntotal[Y]);
    test_assert(jdy >= - ntotal[Y]);
    test_assert(fr >= 0.0);
    test_assert(fr <= 1.0);

    /* The first point in global coordinates involved in the buffer
     * exchange is j1, displaced -(jdy+1) from the local leftmost
     * point. Modular arithmetic ensures 1 <= j1 <= N_y.  */

    jlocal = noffset[Y] + 1;
    j1 = 1 + (jlocal - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];

    test_assert(j1 >= 1);
    test_assert(j1 <= ntotal[Y]);

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
    test_assert(py < cartsz[Y]);
    MPI_Cart_rank(comm, &py, &precv_rank_left);
    py = 1 + (j1 - 1) / nlocal[Y];
    test_assert(py >= 1);
    test_assert(py <= cartsz[Y]);
    MPI_Cart_rank(comm, &py, &precv_rank_right);
  }

  lees_edw_free(le);

  return 0;
}

/*****************************************************************************
 *
 *  test_le_parallel2
 *
 *  Designed for the 4-point interpolation.
 *
 *****************************************************************************/

static int test_le_parallel2(pe_t * pe, cs_t * cs) {

  int n;
  int jdy;
  int jc, j1, j2;
  int n1, n2, n3;
  int nhalo;

  int ntotal[3];
  int nlocal[3];
  int noffset[3];

  double len[3];
  double fr;
  double dy;
  double uy_set = 0.25;

  assert(pe);
  assert(cs);

  /* Check displacement calculations. Run to a displacement which is
   * at least a couple of periodic images. */

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_nhalo(cs, &nhalo);
  cs_ltot(cs, len);

  for (n = -5000; n <= 5000; n++) {

    /* Set the displacement dy, and take modulo L(Y), which should
     * give -L(Y) < dy < +L(Y) */

    dy = uy_set*n;
    dy = fmod(dy, len[Y]);

    test_assert(dy > -len[Y]);
    test_assert(dy < +len[Y]);

    /* The integral part of the displacement jdy and the fractional
     * part are... */

    jdy = floor(dy);
    fr = dy - jdy;

    test_assert(jdy < ntotal[Y]);
    test_assert(jdy >= - ntotal[Y]);
    test_assert(fr >= 0.0);
    test_assert(fr <= 1.0);

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 3 - nhalo + 2*ntotal[Y]) % ntotal[Y];
    j2 = 1 + (j1 - 1) % nlocal[Y];

    test_assert(j2 >= 1);
    test_assert(j2 <= nlocal[Y]);

    n1 = nlocal[Y] - j2 + 1;
    n2 = imin(nlocal[Y], j2 + 2 + 2*nhalo);
    n3 = imax(0, j2 - nlocal[Y] + 2 + 2*nhalo);

    test_assert((n1 + n2 + n3) == nlocal[Y] + 2*nhalo + 3);
  }

  return 0;
}
