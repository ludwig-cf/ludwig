/*****************************************************************************
 *
 *  test_leesedwards.c
 *
 *  Lees Edwards structure
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "physics.h"
#include "util.h"
#include "tests.h"

static int test_parallel1(pe_t * pe, cs_t * cs);
static int test_le_parallel2(pe_t * pe, cs_t * cs);

int test_lees_edw_create(pe_t * pe, cs_t * cs);
int test_lees_edw_buffer_displacement(pe_t * pe, cs_t * cs);
int test_lees_edw_buffer_du(pe_t * pe, cs_t * cs);
int test_lees_edw_buffer_duy(pe_t * pe, cs_t * cs);

int test_lees_edw_type_to_string(void);
int test_lees_edw_type_from_string(void);
int test_lees_edw_opts_to_json(void);
int test_lees_edw_opts_from_json(void);

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

  test_lees_edw_create(pe, cs);
  test_lees_edw_buffer_displacement(pe, cs);
  test_lees_edw_buffer_du(pe, cs);
  test_lees_edw_buffer_duy(pe, cs);

  test_parallel1(pe, cs);
  test_le_parallel2(pe, cs);

  test_lees_edw_type_to_string();
  test_lees_edw_type_from_string();
  test_lees_edw_opts_to_json();
  test_lees_edw_opts_from_json();

  physics_free(phys);
  cs_free(cs);

  pe_info(pe, "PASS     ./unit/test_le\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lees_edw_create
 *
 *****************************************************************************/

int test_lees_edw_create(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* No planes */
  {
    lees_edw_options_t opts = {.nplanes = 0};
    lees_edw_t * le = NULL;

    ifail = lees_edw_create(pe, cs, &opts, &le);
    assert(le);
    assert(lees_edw_nplane_total(le) == 0);
    assert(lees_edw_nplane_local(le) == 0);
    lees_edw_free(le);
  }

  /* Two planes */
  {
    lees_edw_options_t opts = {.nplanes = 2, .type = LE_SHEAR_TYPE_STEADY,
                               .nt0 = 0, .uy = 0.01};
    lees_edw_t * le = NULL;
    ifail = lees_edw_create(pe, cs, &opts, &le);
    assert(ifail == 0);
    assert(lees_edw_nplane_total(le) == 2);
    lees_edw_free(le);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lees_edw_buffer_displacement
 *
 *****************************************************************************/

int test_lees_edw_buffer_displacement(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Steady displacement is uy.t, with a sign dependent on buffer ib */
  {
    int ib = 0;
    double t = 2.0;
    double dy = 0.0;

    lees_edw_options_t opts = {.nplanes = 1, .type = LE_SHEAR_TYPE_STEADY,
                               .nt0 = 0, .uy = 0.01};
    lees_edw_t * le = NULL;

    ifail = lees_edw_create(pe, cs, &opts, &le);
    assert(ifail == 0);
    lees_edw_buffer_displacement(le, ib, t, &dy);
    if (fabs(dy + t*opts.uy) > DBL_EPSILON) ifail = -1; /* ib = 0 is -ve */
    assert(ifail == 0);
    lees_edw_free(le);
  }

  /* Oscillatory displacement */
  /* The uy is just set to fix up the displacement at time t = 0.5;
   * it's  not realistic */
  {
    lees_edw_options_t opts = {.nplanes = 1, .type = LE_SHEAR_TYPE_OSCILLATORY,
                               .period = 2, .nt0 = 0, .uy = 4.0*atan(1.0)};
    lees_edw_t * le = NULL;
    int ib = 0;
    double t = 0.5;
    double dy = 0.0;

    ifail = lees_edw_create(pe, cs, &opts, &le);
    assert(ifail == 0);
    lees_edw_buffer_displacement(le, ib, t, &dy);
    if (fabs(dy - 1.0) > FLT_EPSILON) ifail = -1;
    assert(ifail == 0);
    lees_edw_free(le);
  }

  return ifail;
}
/*****************************************************************************
 *
 *  test_lees_edw_buffer_du
 *
 *****************************************************************************/

int test_lees_edw_buffer_du(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  lees_edw_options_t opts = {.nplanes = 1, .type = LE_SHEAR_TYPE_STEADY,
                             .nt0 = 0, .uy = 0.01};
  lees_edw_t * le = NULL;

  ifail = lees_edw_create(pe, cs, &opts, &le);
  assert(ifail == 0);

  /* This follows duy */
  {
    int ib = 0;
    int nhalo = -1;

    lees_edw_nhalo(le, &nhalo);

    for (int p = 0; p < lees_edw_nplane_local(le); p++) {
      for (int nh = 0; nh < nhalo; nh++) {
	int isgn = -1;
	double u[3] = {-1.0, -1.0, -1.0};
	lees_edw_buffer_du(le, ib, u);
	assert(u[X] == 0.0 && u[Z] == 0.0);
	if (fabs(u[Y] - opts.uy*isgn) > DBL_EPSILON) ifail = -1;
	assert(ifail == 0);
	ib++;
      }
      for (int nh = 0; nh < nhalo; nh++) {
	int isgn = +1;
	double u[3] = {-1.0, -1.0, -1.0};
	lees_edw_buffer_du(le, ib, u);
	assert(u[X] == 0.0 && u[Z] == 0.0);
	if (fabs(u[Y] - opts.uy*isgn) > DBL_EPSILON) ifail = -1;
	assert(ifail == 0);
	ib++;
      }
    }
  }

  lees_edw_free(le);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lees_edw_buffer_duy
 *
 *****************************************************************************/

int test_lees_edw_buffer_duy(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  lees_edw_options_t opts = {.nplanes = 2, .type = LE_SHEAR_TYPE_STEADY,
                             .nt0 = 0, .uy = 0.01};
  lees_edw_t * le = NULL;

  ifail = lees_edw_create(pe, cs, &opts, &le);
  assert(ifail == 0);

  /* Check pattern using alternative initialisation code */

  {
    int nhalo = -1;
    int ib = 0; /* Increments by +1 moving along the buffer */

    lees_edw_nhalo(le, &nhalo);

    for (int p = 0; p < lees_edw_nplane_local(le); p++) {
      for (int nh = 0; nh < nhalo; nh++) {
	assert(lees_edw_buffer_duy(le, ib) == -1);
	ib++;
      }
      for (int nh = 0; nh < nhalo; nh++) {
	assert(lees_edw_buffer_duy(le, ib) == +1);
	ib++;
      }
    }
  }

  lees_edw_free(le);

  return ifail;
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

  lees_edw_options_t myopts = {0};
  lees_edw_options_t * opts = &myopts;
  lees_edw_t * le = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);

  opts->nplanes = nplane;
  opts->uy = uy_set;

  cs_cartsz(cs, cartsz);
  lees_edw_create(pe, cs, opts, &le);

  /* Total number of planes... */
  test_assert(lees_edw_nplane_total(le) == nplane);

  /* Local number of planes... */
  nplane_local = nplane / cartsz[X];
  test_assert(lees_edw_nplane_local(le) == nplane_local);

  lees_edw_plane_uy(le, &uy);
  test_assert(fabs(uy - uy_set) < TEST_DOUBLE_TOLERANCE);


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

/*****************************************************************************
 *
 *  test_lees_edw_type_to_string
 *
 *****************************************************************************/

int test_lees_edw_type_to_string(void) {

  int ifail = 0;

  {
    lees_edw_enum_t mytype = LE_SHEAR_TYPE_INVALID;
    const char * str = lees_edw_type_to_string(mytype);

    if (strcmp(str, "INVALID") != 0) ifail = -1;
    assert(ifail == 0);
  }

  {
    lees_edw_enum_t mytype = LE_SHEAR_TYPE_STEADY;
    const char * str = lees_edw_type_to_string(mytype);

    if (strcmp(str, "STEADY") != 0) ifail = -1;
    assert(ifail == 0);
  }

  {
    lees_edw_enum_t mytype = LE_SHEAR_TYPE_OSCILLATORY;
    const char * str = lees_edw_type_to_string(mytype);

    if (strcmp(str, "OSCILLATORY") != 0) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lees_edw_type_from_string
 *
 *****************************************************************************/

int test_lees_edw_type_from_string(void) {

  int ifail = 0;

  {
    lees_edw_enum_t mytype = lees_edw_type_from_string("RUBBISH");
    if (mytype != LE_SHEAR_TYPE_INVALID) ifail = -1;
    assert(ifail == 0);
  }

  {
    lees_edw_enum_t mytype = lees_edw_type_from_string("STEADY");
    if (mytype != LE_SHEAR_TYPE_STEADY) ifail = -1;
    assert(ifail == 0);
  }

  {
    lees_edw_enum_t mytype = lees_edw_type_from_string("OSCILLATORY");
    if (mytype != LE_SHEAR_TYPE_OSCILLATORY) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lees_edw_opts_to_json
 *
 *****************************************************************************/

int test_lees_edw_opts_to_json(void) {

  int ifail = 0;

  {
    /* No planes. Something of a smoke test here ... */
    cJSON * json = NULL;
    lees_edw_options_t opts = {0};

    ifail = lees_edw_opts_to_json(&opts, &json);
    assert(ifail == 0);

    assert(json);
    cJSON_Delete(json);
  }

  {
    /* Steady case. */
    cJSON * json = NULL;
    lees_edw_options_t opts = {.nplanes = 2,
			    .type    = LE_SHEAR_TYPE_STEADY,
			    .nt0     = 10,
			    .uy      = 0.001};

    ifail = lees_edw_opts_to_json(&opts, &json);
    assert(ifail == 0);

    assert(json);
    cJSON_Delete(json);
  }

  {
    /* Oscillatory case */
    cJSON * json = NULL;
    lees_edw_options_t opts = {.nplanes = 8,
			    .type    = LE_SHEAR_TYPE_OSCILLATORY,
			    .period  = 100,
			    .nt0     = 0,
			    .uy      = 0.02};

    ifail = lees_edw_opts_to_json(&opts, &json);
    assert(ifail == 0);

    assert(json);
    cJSON_Delete(json);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lees_edw_opts_from_json
 *
 *****************************************************************************/

int test_lees_edw_opts_from_json(void) {

  int ifail = 0;

  {
    /* No planes */
    cJSON * json = cJSON_Parse("{\"Number of planes\": 0}");
    lees_edw_options_t opts = {.nplanes = 1};
    ifail = lees_edw_opts_from_json(json, &opts);
    assert(ifail == 0);
    assert(opts.nplanes == 0);

    cJSON_Delete(json);
  }

  {
    /* Steady shear case */
    cJSON * json = cJSON_Parse("{"
			       "\"Number of planes\": 2,"
			       "\"Shear type\":       \"STEADY\","
			       "\"Reference time\":   10,"
			       "\"Plane speed\":      0.001"
                               "}");
    lees_edw_options_t opts = {0};

    ifail = lees_edw_opts_from_json(json, &opts);
    assert(ifail == 0);
    assert(opts.nplanes == 2);
    assert(opts.type    == LE_SHEAR_TYPE_STEADY);
    assert(opts.nt0     == 10);
    assert(fabs(opts.uy - 0.001) < DBL_EPSILON);

    cJSON_Delete(json);
  }

  {
    /* Oscillatory shear case */
    cJSON * json = cJSON_Parse("{"
			       "\"Number of planes\":    8,"
			       "\"Shear type\":          \"OSCILLATORY\","
			       "\"Period (timesteps)\":  100,"
			       "\"Reference time\":      0,"
			       "\"Plane speed\":         0.02"
                               "}");
    lees_edw_options_t opts = {0};

    ifail = lees_edw_opts_from_json(json, &opts);
    assert(ifail == 0);
    assert(opts.nplanes == 8);
    assert(opts.type    == LE_SHEAR_TYPE_OSCILLATORY);
    assert(opts.period  == 100);
    assert(opts.nt0     == 0);
    assert(fabs(opts.uy - 0.02) < DBL_EPSILON);

    cJSON_Delete(json);
  }

  return ifail;
}
