/*****************************************************************************
 *
 *  test_colloids.c
 *
 *  Colloid cell list et al.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "tests.h"

int test_colloids_info_with_ncell(pe_t * pe, cs_t * cs, int ncellref[3]);
int test_colloids_info_add_local(colloids_info_t * cinfo);
int test_colloids_info_cell_coords(colloids_info_t * cinfo);

int test_colloids_info_initialise(pe_t * pe, cs_t * cs);
int test_colloids_info_finalise(pe_t * pe, cs_t * cs);
int test_colloids_array(pe_t * pe, cs_t * cs);


/*****************************************************************************
 *
 *  test_colloids_info_suite
 *
 *****************************************************************************/

int test_colloids_info_suite(void) {

  int ncell[3];
  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_colloids_info_initialise(pe, cs);
  test_colloids_info_finalise(pe, cs);

  /* Older tests */
  ncell[X] = 2;
  ncell[Y] = 2;
  ncell[Z] = 2;

  test_colloids_info_with_ncell(pe, cs, ncell);

  ncell[X] = 3;
  ncell[Y] = 5;
  ncell[Z] = 7;
  test_colloids_info_with_ncell(pe, cs, ncell);

  ncell[X] = 3;
  ncell[Y] = 3;
  ncell[Z] = 3;
  test_colloids_info_with_ncell(pe, cs, ncell);

  ncell[X] = 4;
  ncell[Y] = 6;
  ncell[Z] = 8;
  test_colloids_info_with_ncell(pe, cs, ncell);

  test_colloids_array(pe, cs);

  pe_info(pe, "PASS     ./unit/test_colloids\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_info_initialise
 *
 *****************************************************************************/

int test_colloids_info_initialise(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Options specify no colloids (default options) */

  {
    colloid_options_t options = colloid_options_default();
    colloids_info_t info = {};

    ifail = colloids_info_initialise(pe, cs, &options, &info);
    assert(ifail == 0);

    assert(info.nhalo      == 1);
    assert(info.ntotal     == 0);
    assert(info.nallocated == 0);
    assert(info.ncell[X]   == options.ncell[X]);
    assert(info.ncell[Y]   == options.ncell[Y]);
    assert(info.ncell[Z]   == options.ncell[Z]);
    /* strides */
    /* sites */

    assert(info.nsubgrid     == 0);

    assert(fabs(info.rho0 - 1.0) < DBL_EPSILON);

    assert(info.isgravity    == 0);
    assert(info.isbuoyancy   == 0);
    assert(fabs(info.fgravity[X] - 0.0) < DBL_EPSILON);
    assert(fabs(info.fgravity[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(info.fgravity[Z] - 0.0) < DBL_EPSILON);
    assert(fabs(info.bgravity[X] - 0.0) < DBL_EPSILON);
    assert(fabs(info.bgravity[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(info.bgravity[Z] - 0.0) < DBL_EPSILON);

    assert(info.clist     != NULL);
    assert(info.map_old   == NULL);
    assert(info.map_new   == NULL);
    assert(info.headall   == NULL);
    assert(info.headlocal == NULL);

    assert(info.pe        == pe);
    assert(info.cs        == cs);
    assert(info.target    != NULL);

    colloids_info_finalise(&info);
  }

  /* For have_colloids, check components which are now active */

  {
    colloid_options_t opts = colloid_options_have_colloids(1);
    colloids_info_t info   = {};

    ifail = colloids_info_initialise(pe, cs, &opts, &info);
    assert(ifail == 0);
    assert(info.map_new != NULL);
    assert(info.map_old != NULL);

    colloids_info_finalise(&info);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_info_finalise
 *
 *****************************************************************************/

int test_colloids_info_finalise(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  {
    colloid_options_t opts = colloid_options_default();
    colloids_info_t   info = {};

    ifail = colloids_info_initialise(pe, cs, &opts, &info);
    ifail = colloids_info_finalise(&info);
    assert(ifail == 0);

    /* A subset of components... */
    assert(info.nhalo  == 0);
    assert(info.clist  == NULL);
    assert(info.target == NULL);
  }

  return ifail;
}


/*****************************************************************************
 *
 *  test_colloids_info_with_ncell
 *
 *****************************************************************************/

int test_colloids_info_with_ncell(pe_t * pe, cs_t * cs, int ncellref[3]) {

  int ia;
  int ncell[3] = {0, 0, 0};
  int mpi_cartsz[3];

  double ltot[3];
  double lcell[3];
  double lcellref;

  colloid_options_t opts  = colloid_options_ncell(ncellref);
  colloids_info_t * cinfo = NULL;

  assert(pe);
  assert(cs);

  cs_ltot(cs, ltot);
  cs_cartsz(cs, mpi_cartsz);

  colloids_info_create(pe, cs, &opts, &cinfo);
  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);

  test_assert(ncell[X] == ncellref[X]);
  test_assert(ncell[Y] == ncellref[Y]);
  test_assert(ncell[Z] == ncellref[Z]);

  colloids_info_lcell(cinfo, lcell);

  for (ia = 0; ia < 3; ia++) {
    lcellref = ltot[ia] / (mpi_cartsz[ia]*ncellref[ia]);
    test_assert(fabs(lcell[ia] - lcellref) < TEST_DOUBLE_TOLERANCE);
  }

  /* Longer tests */

  test_colloids_info_cell_coords(cinfo);
  test_colloids_info_add_local(cinfo);

  colloids_info_free(&cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_info_add_local
 *
 *****************************************************************************/

int test_colloids_info_add_local(colloids_info_t * cinfo) {

  int index;
  int ncount;
  int ncolloid;
  int noffset[3];
  int icell[3];
  double r[3];
  double lmin[3];

  double a0 = 2.3;
  double ah = 2.3;
  colloid_state_t s = {};

  colloid_t * pcref = NULL;
  colloid_t * pc = NULL;

  assert(cinfo);

  cs_lmin(cinfo->cs, lmin);
  cs_nlocal_offset(cinfo->cs, noffset);

  index = 1 + pe_mpi_rank(cinfo->pe);

  /* This should not go in locally */

  r[X] = lmin[X] + 1.0*(noffset[X] - 1);
  r[Y] = lmin[Y] + 1.0*(noffset[Y] - 1);
  r[Z] = lmin[Z] + 1.0*(noffset[Z] - 1);

  colloid_state_init_sphere(index, a0, ah, r, &s);
  colloids_info_add_local(cinfo, &s, &pcref);
  assert(pcref == NULL);

  /* This one will, giving one colloid per MPI task */

  r[X] = lmin[X] + 1.0*(noffset[X] + 1);
  r[Y] = lmin[Y] + 1.0*(noffset[Y] + 1);
  r[Z] = lmin[Z] + 1.0*(noffset[Z] + 1);

  colloid_state_init_sphere(index, a0, ah, r, &s);
  colloids_info_add_local(cinfo, &s, &pcref);
  assert(pcref != NULL);

  colloids_info_nlocal(cinfo, &ncolloid);
  test_assert(ncolloid == 1);

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &ncolloid);

  /* Check the colloid is in the cell */

  colloids_info_cell_coords(cinfo, r, icell);
  colloids_info_cell_count(cinfo, icell[X], icell[Y], icell[Z], &ncount);
  test_assert(ncount == 1);

  colloids_info_cell_list_head(cinfo, icell[X], icell[Y], icell[Z], &pc);
  test_assert(pc == pcref);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_info_cell_coords
 *
 *****************************************************************************/

int test_colloids_info_cell_coords(colloids_info_t * cinfo) {

  int ncell[3];
  int icell[3];
  int nlocal[3];
  int noffset[3];
  double r[3];
  double lcell[3];
  double lmin[3];
  double delta = FLT_EPSILON;

  assert(cinfo);

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, noffset);
  cs_lmin(cinfo->cs, lmin);

  colloids_info_ncell(cinfo, ncell);
  colloids_info_lcell(cinfo, lcell);

  /* Start in local cell [1,1,1] */

  r[X] = lmin[X] + 1.0*noffset[X] + 0.5*delta;
  r[Y] = lmin[Y] + 1.0*noffset[Y] + 0.5*delta;
  r[Z] = lmin[Z] + 1.0*noffset[Z] + 0.5*delta;

  colloids_info_cell_coords(cinfo, r, icell);
  test_assert(icell[X] == 1);
  test_assert(icell[Y] == 1);
  test_assert(icell[Z] == 1);

  /* Translate to [0,0,0] */

  r[X] -= lcell[X];
  r[Y] -= lcell[Y];
  r[Z] -= lcell[Z];

  colloids_info_cell_coords(cinfo, r, icell);
  test_assert(icell[X] == 0);
  test_assert(icell[Y] == 0);
  test_assert(icell[Z] == 0);

  /* Move two cells up to [2,2,2] */

  r[X] += 2.0*lcell[X];
  r[Y] += 2.0*lcell[Y];
  r[Z] += 2.0*lcell[Z];

  colloids_info_cell_coords(cinfo, r, icell);
  test_assert(icell[X] == 2);
  test_assert(icell[Y] == 2);
  test_assert(icell[Z] == 2);

  /* Now, shave a little off the position and we should get back
   * to [1,1,1] */

  r[X] -= delta;
  r[Y] -= delta;
  r[Z] -= delta;

  colloids_info_cell_coords(cinfo, r, icell);
  test_assert(icell[X] == 1);
  test_assert(icell[Y] == 1);
  test_assert(icell[Z] == 1);

  /* And this should catapult us to the last cell in each direction
   * in the halo region */

  r[X] += 1.0*nlocal[X];
  r[Y] += 1.0*nlocal[Y];
  r[Z] += 1.0*nlocal[Z];

  colloids_info_cell_coords(cinfo, r, icell);

  test_assert(icell[X] == ncell[X] + 1);
  test_assert(icell[Y] == ncell[Y] + 1);
  test_assert(icell[Z] == ncell[Z] + 1);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_array
 *
 *****************************************************************************/

int test_colloids_array(pe_t * pe, cs_t * cs) {
  colloids_info_t *cinfo = NULL;
  colloid_t *pc = NULL;
  colloid_state_t s = {};

  double r[3];
  double lmin[3];
  double delta = FLT_EPSILON;
  int noffset[3];
  int index;
  double a0 = 2.3;
  double ah = 2.3;
  int n_colloids = 3;

  cs_nlocal_offset(cs, noffset);
  cs_lmin(cs, lmin);
  
  colloid_options_t options = colloid_options_default();

  options.have_colloids = 1;
  colloids_info_create(pe, cs, &options, &cinfo);


  for (int i = 0; i < n_colloids; i++) {
    r[X] = lmin[X] + 1.0*noffset[X] + 0.5*delta + i*0.1*delta;
    r[Y] = lmin[Y] + 1.0*noffset[Y] + 0.5*delta + i*0.1*delta;
    r[Z] = lmin[Z] + 1.0*noffset[Z] + 0.5*delta + i*0.1*delta;
  
    index = 1 + pe_mpi_rank(cinfo->pe) + i;

    colloid_state_init_sphere(index, a0, ah, r, &s);

    colloids_info_add_local(cinfo, &s, &pc);

    update_colloids_array(cinfo);

    colloids_array_check(cinfo);
  }
  colloids_info_free(&cinfo);

  return 0;
}