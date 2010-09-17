/*****************************************************************************
 *
 *  test_colloid_sums.c
 *
 *  Test of the various sum routines.
 *
 *  $Id: test_colloid_sums.c,v 1.1.2.1 2010-09-17 16:36:25 kevin Exp $
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
#include "colloids_halo.h"
#include "colloid_sums.h"
#include "tests.h"

static void test_colloid_sums_reference_set(colloid_t * cref);
static void test_colloid_sums_copy(colloid_t ref, colloid_t * pc);
static void test_colloid_sums_assert(colloid_t c1, colloid_t * c2);
static void test_colloid_sums_edge_x(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3] = {1024, 512, 256};

  pe_init(argc, argv);

  coords_ntotal_set(ntotal);
  coords_init();

  test_colloid_sums_edge_x();

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_edge_x
 *
 *  Place a single particle at {delta, Ly/2, Lz/2} to test
 *  the x communication.
 *
 *  We use 4 cells to prevent copies in directions other than x
 *  when a call to colloids_halo_state() is made.
 *
 *****************************************************************************/

static void test_colloid_sums_edge_x(void) {

  int         index;
  int         ncell[3] = {4, 4, 4};
  int         ic, jc, kc;

  double      r0[3];
  colloid_t * pc;
  colloid_t * pctest;
  colloid_t   cref;   /* All ranks get the same reference colloid */

  test_colloid_sums_reference_set(&cref);
  colloids_cell_ncell_set(ncell);

  colloids_init();

  index = 1;
  r0[X] = Lmin(X) + 0.5;
  r0[Y] = Lmin(Y) + 0.5*L(Y);
  r0[Z] = Lmin(Z) + 0.5*L(Z);

  pc = colloid_add_local(index, r0);

  if (pc) {
    /* The 'owner' sets some values for the sum */
    test_colloid_sums_copy(cref, pc);
  }

  colloids_halo_state();
  colloid_sums_dim(X, COLLOID_SUM_STRUCTURE);

  /* Everywhere check colloid index = 1 has the correct sum */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	pctest = colloids_cell_list(ic, jc, kc);

	if (pctest) {
	  /* Check the totals */
	  test_colloid_sums_assert(cref, pctest);

	  /* Should be at most one colloid */
	  test_assert(pctest->next == NULL);
	}
	/* Next cell */
      }
    }
  }

  /* Finish */

  colloids_finish();

  return;
}

/*****************************************************************************
 *
 *  test_colloid_sums_reference_set
 *
 *****************************************************************************/

static void test_colloid_sums_reference_set(colloid_t * pc) {

  int ia;
  int ivalue;

  ivalue = 2;

  pc->sumw = 1.0*ivalue++;

  for (ia = 0; ia < 3; ia++) {
    pc->cbar[ia] = 1.0*ivalue++;
    pc->rxcbar[ia] = 1.0*ivalue++;
  }
 
  return;
}

/*****************************************************************************
 *
 *  test_colloid_sums_copy
 *
 *  Copy the sum information from reference to pc.
 *
 *****************************************************************************/

static void test_colloid_sums_copy(colloid_t ref, colloid_t * pc) {

  int ia;

  pc->sumw = ref.sumw;

  for (ia = 0; ia < 3; ia++) {
    pc->cbar[ia] = ref.cbar[ia];
    pc->rxcbar[ia] = ref.rxcbar[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  test_colloid_sums_assert
 *
 *  Assert that two colloids hold the same summed quantities.
 *
 *****************************************************************************/

static void test_colloid_sums_assert(colloid_t c1, colloid_t * c2) {

  int ia;

  test_assert(fabs(c1.sumw - c2->sumw) < TEST_DOUBLE_TOLERANCE);

  for (ia = 0; ia < 3; ia++) {

    test_assert(fabs(c1.cbar[ia] - c2->cbar[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.rxcbar[ia] - c2->rxcbar[ia]) < TEST_DOUBLE_TOLERANCE);

  }

  return;
}
