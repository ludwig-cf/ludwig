/*****************************************************************************
 *
 *  test_colloids.c
 *
 *  Colloid cell list et al.
 *
 *  $Id: test_colloids.c,v 1.1.2.3 2010-10-08 12:09:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "tests.h"

static void test_colloids_ncell(void);
static void test_colloids_lcell(void);
static void test_colloids_allocate(void);
static void test_colloids_add_local(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ncell[3];

  pe_init(argc, argv);
  coords_init();

  /* The default local number of cells is {2, 2, 2} */

  colloids_cell_ncell(ncell);

  test_assert(ncell[X] == 2);
  test_assert(ncell[Y] == 2);
  test_assert(ncell[Z] == 2);
  test_assert(colloid_ntotal() == 0);

  test_colloids_ncell();
  test_colloids_lcell();
  test_colloids_allocate();
  test_colloids_add_local();

  info("Completed colloids test\n");

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_ncell
 *
 *****************************************************************************/

static void test_colloids_ncell(void) {

  int ncellref[3] = {4, 5, 3};
  int ncell[3];

  colloids_cell_ncell_set(ncellref);
  colloids_cell_ncell(ncell);

  test_assert(ncell[X] == ncellref[X]);
  test_assert(ncell[Y] == ncellref[Y]);
  test_assert(ncell[Z] == ncellref[Z]);

  colloids_init();
  colloids_finish();

  /* Check defaults are restored */

  colloids_cell_ncell(ncell);

  test_assert(ncell[X] == 2);
  test_assert(ncell[Y] == 2);
  test_assert(ncell[Z] == 2);

  return;
}

/*****************************************************************************
 *
 *  test_colloids_lcell
 *
 *****************************************************************************/

static void test_colloids_lcell(void) {

  int ia;
  int ncellref[3] = {3, 4, 5};
  double lcell;
  double lcellref;
  
  colloids_cell_ncell_set(ncellref);

  for (ia = 0; ia < 3; ia++) {
    lcellref = L(ia) / (cart_size(ia)*ncellref[ia]);
    lcell = colloids_lcell(ia);

    test_assert(fabs(lcell - lcellref) < TEST_DOUBLE_TOLERANCE);
  }

  return;
}

/*****************************************************************************
 *
 *  test_colloids_allocate
 *
 *****************************************************************************/

static void test_colloids_allocate(void) {

  colloid_t * pc;

  pc = colloid_allocate();
  test_assert(pc != NULL);
  test_assert(colloids_nalloc() == 1);

  colloid_free(pc);
  test_assert(colloids_nalloc() == 0);

  return;
}

/*****************************************************************************
 *
 *  test_colloids_add_local
 *
 *  Every process adds one colloid locally.
 *
 *****************************************************************************/

static void test_colloids_add_local(void) {

  int index;
  int noffset[3];
  int icell[3];
  double r[3];

  colloid_t * pc;

  coords_nlocal_offset(noffset);
  colloids_init();

  index = 1 + pe_rank();

  /* This should not go in */
  r[X] = Lmin(X) + 1.0*(noffset[X] - 1);
  r[Y] = Lmin(Y) + 1.0*(noffset[Y] - 1);
  r[Z] = Lmin(Z) + 1.0*(noffset[Z] - 1);

  pc = colloid_add_local(index, r);
  test_assert(pc == NULL);
  test_assert(colloids_nalloc() == 0);

  /* This one will */
  r[X] = Lmin(X) + 1.0*(noffset[X] + 1);
  r[Y] = Lmin(Y) + 1.0*(noffset[Y] + 1);
  r[Z] = Lmin(Z) + 1.0*(noffset[Z] + 1);

  pc = colloid_add_local(index, r);
  test_assert(pc != NULL);
  test_assert(colloids_nalloc() == 1);
  test_assert(colloid_nlocal() == 1);

  colloids_ntotal_set();
  test_assert(colloid_ntotal() == pe_size());

  /* Check the cell */

  colloids_cell_coords(r, icell);
  test_assert(colloids_cell_count(icell[X], icell[Y], icell[Z]) == 1);
  test_assert(colloids_cell_list(icell[X], icell[Y], icell[Z]) == pc);

  colloids_finish();
  test_assert(colloids_nalloc() == 0);
  test_assert(colloid_ntotal() == 0);

  return;
}
