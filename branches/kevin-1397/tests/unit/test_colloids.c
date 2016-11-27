/*****************************************************************************
 *
 *  test_colloids.c
 *
 *  Colloid cell list et al.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "colloids_s.h"
#include "tests.h"

int test_colloids_info_with_ncell(pe_t * pe, cs_t * cs, int ncellref[3]);
int test_colloids_info_add_local(colloids_info_t * cinfo);
int test_colloids_info_cell_coords(colloids_info_t * cinfo);

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

  pe_info(pe, "PASS     ./unit/test_colloids\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_info_with_ncell
 *
 *****************************************************************************/

int test_colloids_info_with_ncell(pe_t * pe, cs_t * cs, int ncellref[3]) {

  int ia;
  int ncell[3] = {0, 0, 0};
  double lcell[3];
  double lcellref;
  colloids_info_t * cinfo = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, ncellref, &cinfo);
  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);

  test_assert(ncell[X] == ncellref[X]);
  test_assert(ncell[Y] == ncellref[Y]);
  test_assert(ncell[Z] == ncellref[Z]);

  colloids_info_lcell(cinfo, lcell);

  for (ia = 0; ia < 3; ia++) {
    lcellref = L(ia) / (cart_size(ia)*ncellref[ia]);
    test_assert(fabs(lcell[ia] - lcellref) < TEST_DOUBLE_TOLERANCE);
  }

  /* Longer tests */

  test_colloids_info_cell_coords(cinfo);
  test_colloids_info_add_local(cinfo);

  colloids_info_free(cinfo);

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

  colloids_info_add_local(cinfo, index, r, &pcref);
  test_assert(pcref == NULL);

  /* This one will, giving one colloid per MPI task */

  r[X] = lmin[X] + 1.0*(noffset[X] + 1);
  r[Y] = lmin[Y] + 1.0*(noffset[Y] + 1);
  r[Z] = lmin[Z] + 1.0*(noffset[Z] + 1);

  colloids_info_add_local(cinfo, index, r, &pcref);
  test_assert(pcref != NULL);
  colloids_info_nlocal(cinfo, &ncolloid);
  test_assert(ncolloid == 1);

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &ncolloid);
  test_assert(ncolloid == pe_size());

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
  /* verbose("A cell %d %d %d\n", icell[X], icell[Y], icell[Z]);*/
  test_assert(icell[X] == 1);
  test_assert(icell[Y] == 1);
  test_assert(icell[Z] == 1);

  /* Translate to [0,0,0] */

  r[X] -= lcell[X];
  r[Y] -= lcell[Y];
  r[Z] -= lcell[Z];

  colloids_info_cell_coords(cinfo, r, icell);
  /* verbose("B cell %d %d %d\n", icell[X], icell[Y], icell[Z]);*/
  test_assert(icell[X] == 0);
  test_assert(icell[Y] == 0);
  test_assert(icell[Z] == 0);

  /* Move two cells up to [2,2,2] */

  r[X] += 2.0*lcell[X];
  r[Y] += 2.0*lcell[Y];
  r[Z] += 2.0*lcell[Z];

  colloids_info_cell_coords(cinfo, r, icell);
  /* verbose("C cell %d %d %d\n", icell[X], icell[Y], icell[Z]);*/
  test_assert(icell[X] == 2);
  test_assert(icell[Y] == 2);
  test_assert(icell[Z] == 2);

  /* Now, shave a little off the position and we should get back
   * to [1,1,1] */

  r[X] -= delta;
  r[Y] -= delta;
  r[Z] -= delta;

  colloids_info_cell_coords(cinfo, r, icell);
  /* verbose("D cell %d %d %d\n", icell[X], icell[Y], icell[Z]);*/
  test_assert(icell[X] == 1);
  test_assert(icell[Y] == 1);
  test_assert(icell[Z] == 1);

  /* And this should catapult us to the last cell in each direction
   * in the halo region */

  r[X] += 1.0*nlocal[X];
  r[Y] += 1.0*nlocal[Y];
  r[Z] += 1.0*nlocal[Z];

  colloids_info_cell_coords(cinfo, r, icell);
  /* verbose("E cell %d %d %d\n\n", icell[X], icell[Y], icell[Z]);*/

  test_assert(icell[X] == ncell[X] + 1);
  test_assert(icell[Y] == ncell[Y] + 1);
  test_assert(icell[Z] == ncell[Z] + 1);

  return 0;
}
