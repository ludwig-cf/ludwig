/*****************************************************************************
 *
 *  unit_leesedwards.c
 *
 *  Lees Edwards structure
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2104 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "util.h"
#include "coords.h"
#include "leesedwards.h"
#include "unit_control.h"

int do_test_le1(control_t * ctrl);
int do_test_le2(control_t * ctrl);
int do_test_le_interp3(control_t * ctrl);
int do_test_le_interp4(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_lees_edwards
 *
 *****************************************************************************/

int do_ut_lees_edwards(control_t * ctrl) {

  assert(ctrl);

  do_test_le1(ctrl);
  do_test_le2(ctrl);
  do_test_le_interp3(ctrl);
  do_test_le_interp4(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_le1
 *
 *  Lees Edwards structure.
 *
 *****************************************************************************/

int do_test_le1(control_t * ctrl) {

  int nplane = 8;
  int nplane_local;
  int n, nx, ix;
  int nptotal, nplocal;
  int nlocal[3];
  int cartsz[3];

  double uy0 = 0.08;
  double uy;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  le_t * le = NULL;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Lees Edwards plane structure\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  coords_cartsz(cs, cartsz);

  le_create(cs, &le);
  le_nplane_set(le, nplane);
  le_plane_uy_set(le, uy0);
  le_commit(le);

  try {

    le_nplane_total(le, &nptotal);
    le_nplane_local(le, &nplocal);
    le_plane_uy(le, &uy);

    control_verb(ctrl, "Total number of planes: %d\n", nplane);
    control_macro_test(ctrl, nptotal == nplane);

    nplane_local = nplane / cartsz[X];
    control_verb(ctrl, "Local number of planes: %d\n", nplane_local);
    control_macro_test(ctrl, nplocal == nplane_local);

    control_verb(ctrl, "Plane maximum velocity: %10.4f\n", uy0);
    control_macro_test_dbl_eq(ctrl, uy, uy0, DBL_EPSILON);

    /* Plane locations */

    coords_nlocal(cs, nlocal);

    for (n = 0; n < nplane_local; n++) {
      nx = nlocal[X]/(2*nplane_local) + n*nlocal[X]/nplane_local;
      ix = le_plane_location(le, n);
      control_verb(ctrl, "Plane %d integer position: %d (%d)\n", n, nx, ix);
      control_macro_test(ctrl, nx == ix);
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    le_free(le);
    coords_free(cs);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_le2
 *
 *  Lees Edwards sites, buffer quantites.
 *
 *****************************************************************************/

int do_test_le2(control_t * ctrl) {

  int nplane_ref = 4;
  int nplane;
  int nlocal[3];
  int nhalo;
  int nh2;              /* 2*nhalo */
  int nxb;              /* Number buffer planes in x */
  int nsites;           /* Total lattice sites */
  int nexpect;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  le_t * le = NULL;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Lees Edwards buffers\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  coords_nhalo(cs, &nhalo);
  nh2 = 2*nhalo;
  coords_nlocal(cs, nlocal);

  le_create(cs, &le);
  le_nplane_set(le, nplane_ref);
  le_commit(le);

  try {
    le_nxbuffer(le, &nxb);
    le_nplane_local(le, &nplane);

    nexpect = nh2*nplane;
    control_verb(ctrl, "Lees buffer size: %d (d)\n", nxb, nexpect);
    control_macro_test(ctrl, nxb == nexpect);

    le_nsites(le, &nexpect);
    nsites = (nlocal[X] + nh2 + nxb)*(nlocal[Y] + nh2)*(nlocal[Z] + nh2);
    control_verb(ctrl, "Lees nsites: %d (%d)\n", nsites, nexpect);
    control_macro_test(ctrl, nsites == nexpect);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    le_free(le);
    coords_free(cs);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_le_interp3
 *
 *  Test logic used to compute 3-point interpolation
 *
 *****************************************************************************/

int do_test_le_interp3(control_t * ctrl) {

  const int nplane = 2;
  int n;
  int jdy;
  int jlocal, j1, j1mod;
  int n1, n2;
  int py;
  int precv_rank_left, precv_rank_right;

  int nlocal[3];
  int ntotal[3];
  int noffset[3];
  int cartsz[3];

  const double uy_set = 0.25;
  double fr;
  double dy;
  double ltot[3];

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  le_t * le = NULL;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Lees Edwards 3-point interpolation\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  coords_ltot(cs, ltot);
  coords_cartsz(cs, cartsz);

  le_create(cs, &le);
  le_nplane_set(le, nplane);
  le_plane_uy_set(le, uy_set);
  le_commit(le);

  try {

    /* Check displacement calculations. Run to a displacement which is
     * at least a couple of periodic images. */

    coords_ntotal(cs, ntotal);
    coords_nlocal(cs, nlocal);
    coords_nlocal_offset(cs, noffset);

    le_comm(le, &comm);

    for (n = -5000; n <= 5000; n++) {

      /* Set the displacement dy, and take modulo L(Y), which should
       * give -L(Y) < dy < +L(Y) */

      dy = uy_set*n;
      dy = fmod(dy, ltot[Y]);

      control_macro_test(ctrl, dy > -ltot[Y]);
      control_macro_test(ctrl, dy < +ltot[Y]);

      /* The integral part of the displacement jdy and the fractional
       * part are... */

      jdy = floor(dy);
      fr = dy - jdy;

      control_macro_test(ctrl, jdy < ntotal[Y]);
      control_macro_test(ctrl, jdy < ntotal[Y]);
      control_macro_test(ctrl, jdy >= -ntotal[Y]);
      control_macro_test(ctrl, fr >= 0.0);
      control_macro_test(ctrl, fr <= 1.0);

      /* The first point in global coordinates involved in the buffer
       * exchange is j1, displaced -(jdy+1) from the local leftmost
       * point. Modular arithmetic ensures 1 <= j1 <= N_y.  */

      jlocal = noffset[Y] + 1;
      j1 = 1 + (jlocal - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];

      control_macro_test(ctrl, j1 >= 1);
      control_macro_test(ctrl, j1 <= ntotal[Y]);

      /* The corresponding local coordinate is j1mod */

      j1mod = 1 + j1 % nlocal[Y];

      control_macro_test(ctrl, j1mod >= 1);
      control_macro_test(ctrl, j1mod <= nlocal[Y]);

      /* Number of points we can grab from the left processor is n1 */

      n1 = nlocal[Y] - j1mod + 1;

      control_macro_test(ctrl, n1 >= 1);
      control_macro_test(ctrl, n1 <= nlocal[Y]);

      /* Number of points to grab from right procesor is n2 which is
       * (nlocal[Y] + 1) - n1 */

      n2 = j1mod;

      control_macro_test(ctrl, n2 >= 1);
      control_macro_test(ctrl, n2 <= nlocal[Y]);
      control_macro_test(ctrl, (n1 + n2) == (nlocal[Y] + 1));

      py = (j1 - 1) / nlocal[Y];
      control_macro_test(ctrl, py >= 0);
      control_macro_test(ctrl, py < cartsz[Y]);

      MPI_Cart_rank(comm, &py, &precv_rank_left);
      py = 1 + (j1 - 1) / nlocal[Y];
      control_macro_test(ctrl, py >= 1);
      control_macro_test(ctrl, py <= cartsz[Y]);
      MPI_Cart_rank(comm, &py, &precv_rank_right);
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    le_free(le);
    coords_free(cs);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_le_interp4
 *
 *  Test logic for the 4-point interpolation.
 *
 *****************************************************************************/

int do_test_le_interp4(control_t * ctrl) {

  const int nplane = 2;
  const double uy_set = 0.25;

  int n;
  int jdy;
  int jc, j1, j2;
  int n1, n2, n3;
  int nhalo;

  int ntotal[3];
  int nlocal[3];
  int noffset[3];

  double fr;
  double dy;
  double ltot[3];

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  le_t * le = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Lees-Edwards 4-point interpolation\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  le_create(cs, &le);
  le_nplane_set(le, nplane);
  le_plane_uy_set(le, uy_set);
  le_commit(le);

  /* Check displacement calculations. Run to a displacement which is
   * at least a couple of periodic images. */

  coords_ntotal(cs, ntotal);
  coords_nlocal(cs, nlocal);
  coords_nlocal_offset(cs, noffset);
  coords_nhalo(cs, &nhalo);
  coords_ltot(cs, ltot);

  try {
    for (n = -5000; n <= 5000; n++) {

      /* Set the displacement dy, and take modulo L(Y), which should
       * give -L(Y) < dy < +L(Y) */

      dy = uy_set*n;
      dy = fmod(dy, ltot[Y]);

      control_macro_test(ctrl, dy > -ltot[Y]);
      control_macro_test(ctrl, dy < +ltot[Y]);

      /* The integral part of the displacement jdy and the fractional
       * part are... */

      jdy = floor(dy);
      fr = dy - jdy;

      control_macro_test(ctrl, jdy <   ntotal[Y]);
      control_macro_test(ctrl, jdy >= -ntotal[Y]);
      control_macro_test(ctrl, fr >= 0.0);
      control_macro_test(ctrl, fr <= 1.0);

      jc = noffset[Y] + 1;
      j1 = 1 + (jc - jdy - 3 - nhalo + 2*ntotal[Y]) % ntotal[Y];
      j2 = 1 + (j1 - 1) % nlocal[Y];

      control_macro_test(ctrl, j2 >= 1);
      control_macro_test(ctrl, j2 <= nlocal[Y]);

      n1 = nlocal[Y] - j2 + 1;
      n2 = imin(nlocal[Y], j2 + 2 + 2*nhalo);
      n3 = imax(0, j2 - nlocal[Y] + 2 + 2*nhalo);

      control_macro_test(ctrl, (n1 + n2 + n3) == nlocal[Y] + 2*nhalo + 3);
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    le_free(le);
    coords_free(cs);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}
