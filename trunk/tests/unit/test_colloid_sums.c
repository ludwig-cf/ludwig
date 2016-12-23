/*****************************************************************************
 *
 *  test_colloid_sums.c
 *
 *  Test of the various sum routines.
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
#include "colloids.h"
#include "colloids_halo.h"
#include "colloid_sums.h"
#include "tests.h"

static int dim_; /* Current direction */

static int test_colloid_sums_1d(pe_t * pe);
static int test_colloid_sums_reference_set(colloid_t * cref, int seed);
static int test_colloid_sums_copy(colloid_t ref, colloid_t * pc);
static int test_colloid_sums_assert(colloid_t c1, colloid_t * c2);
static int test_colloid_sums_edge(pe_t * pe, cs_t * cs, int ncell[3],
				  const double r0[3]);
static int test_colloid_sums_move(pe_t * pe);
static int test_colloid_sums_conservation(pe_t * pe);

/*****************************************************************************
 *
 *  test_colloid_sums_suite
 *
 *****************************************************************************/

int test_colloid_sums_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_colloid_sums_1d(pe);
  test_colloid_sums_move(pe);
  test_colloid_sums_conservation(pe);

  pe_info(pe, "PASS     ./unit/test_colloid_sums\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_1d
 *
 *  Place one colloid locally at the centre of the domain.
 *  In each direction in turn, we put it close to the (periodic) boundary.
 *
 *****************************************************************************/

static int test_colloid_sums_1d(pe_t * pe) {

  int ntotal[3] = {1024, 512, 256};
  int nlocal[3];
  int ncell[3];

  double r0[3];
  cs_t * cs = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  /* We use cells > 2 to prevent copies in directions other than dim_
   * when a call to colloids_halo_state() is made. */

  cs_nlocal(cs, nlocal);

  dim_ = X;
  ncell[X] = 4;
  ncell[Y] = 3;
  ncell[Z] = 3;

  r0[X] = Lmin(X) + 0.5;
  r0[Y] = Lmin(Y) + 0.5*nlocal[Y];
  r0[Z] = Lmin(Z) + 0.5*nlocal[Z];

  test_colloid_sums_edge(pe, cs, ncell, r0);

  ncell[X] = 2;
  ncell[Y] = 4;
  ncell[Z] = 3;

  test_colloid_sums_edge(pe, cs, ncell, r0);

  dim_ = Y;
  r0[X] = Lmin(X) + 0.5*nlocal[X];
  r0[Y] = Lmin(Y) + 0.5;
  r0[Z] = Lmin(Z) + 0.5*nlocal[Z];

  test_colloid_sums_edge(pe, cs, ncell, r0);

  dim_ = Z;
  r0[X] = Lmin(X) + 0.5*nlocal[X];
  r0[Y] = Lmin(Y) + 0.5*nlocal[Y];
  r0[Z] = Lmin(Z) + 0.5;

  test_colloid_sums_edge(pe, cs, ncell, r0);

  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_edge
 *
 *  Place a single particle at r0 to test the communication.
 *
 *****************************************************************************/

static int test_colloid_sums_edge(pe_t * pe, cs_t * cs, int ncell[3],
				  const double r0[3]) {
  int index;
  int ic, jc, kc;

  colloid_t * pc = NULL;
  colloid_t   cref1;   /* All ranks get the same reference colloids */
  colloid_t   cref2;
  colloid_sum_t * halosum = NULL;
  colloids_info_t * cinfo = NULL;

  test_colloid_sums_reference_set(&cref1, 1);
  test_colloid_sums_reference_set(&cref2, 2);

  colloids_info_create(pe, cs, ncell, &cinfo);
  assert(cinfo);
  colloid_sums_create(cinfo, &halosum);
  assert(halosum);

  /* This must work in parallel to initialise only a single particle
   * which only gets swapped in the x-direction. */

  index = 1;
  colloids_info_add_local(cinfo, index, r0, &pc);
  if (pc) {
    test_colloid_sums_copy(cref1, pc);
  }

  index = 2;
  colloids_info_add_local(cinfo, index, r0, &pc);
  if (pc) {
    test_colloid_sums_copy(cref2, pc);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  colloids_halo_state(cinfo);
  colloid_sums_1d(halosum, X, COLLOID_SUM_STRUCTURE);
  colloid_sums_1d(halosum, X, COLLOID_SUM_DYNAMICS);
  colloid_sums_1d(halosum, X, COLLOID_SUM_ACTIVE);

  if (dim_ == Y || dim_ == Z) {
    colloid_sums_1d(halosum, Y, COLLOID_SUM_STRUCTURE);
    colloid_sums_1d(halosum, Y, COLLOID_SUM_DYNAMICS);
    colloid_sums_1d(halosum, Y, COLLOID_SUM_ACTIVE);
  }

  if (dim_ == Z) {
    colloid_sums_1d(halosum, Z, COLLOID_SUM_STRUCTURE);
    colloid_sums_1d(halosum, Z, COLLOID_SUM_DYNAMICS);
    colloid_sums_1d(halosum, Z, COLLOID_SUM_ACTIVE);
  }

  /* Everywhere check colloid index = 1 has the correct sum */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	if (pc) {
	  /* Check the totals */
	  if (pc->s.index == 1) test_colloid_sums_assert(cref1, pc);
	  if (pc->s.index == 2) test_colloid_sums_assert(cref2, pc);
	}
	/* Next cell */
      }
    }
  }

  /* Finish */

  colloid_sums_free(halosum);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_reference_set
 *
 *****************************************************************************/

static int test_colloid_sums_reference_set(colloid_t * pc, int seed) {

  int ia;
  int ivalue;

  assert(pc);

  ivalue = seed;

  /* STRUCTURE MESSAGE TYPE */
  /* Note, we haven't included s.deltaphi as this is part of the
   * state, which is involved in the halo swap, as well as the sum. */

  pc->sumw = 1.0*ivalue++;
  pc->deltam = 1.0*ivalue++;

  for (ia = 0; ia < 3; ia++) {
    pc->cbar[ia] = 1.0*ivalue++;
    pc->rxcbar[ia] = 1.0*ivalue++;
  }
 
  /* DYNAMICS */

  for (ia = 0; ia < 3; ia++) {
    pc->f0[ia] = 1.0*ivalue++;
    pc->t0[ia] = 1.0*ivalue++;
    pc->force[ia] = 1.0*ivalue++;
    pc->torque[ia] = 1.0*ivalue++;
  }

  pc->sump = 1.0*ivalue++;

  for (ia = 0; ia < 21; ia++) {
    pc->zeta[ia] = 1.0*ivalue++;
  }

  /* ACTIVE */

  for (ia = 0; ia < 3; ia++) {
    pc->fc0[ia] = 1.0*ivalue++;
    pc->tc0[ia] = 1.0*ivalue++;
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_copy
 *
 *  Copy the sum information from reference to pc.
 *
 *****************************************************************************/

static int test_colloid_sums_copy(colloid_t ref, colloid_t * pc) {

  int ia;

  assert(pc);

  pc->sumw = ref.sumw;
  pc->sump = ref.sump;

  for (ia = 0; ia < 3; ia++) {
    pc->cbar[ia] = ref.cbar[ia];
    pc->rxcbar[ia] = ref.rxcbar[ia];
    pc->f0[ia] = ref.f0[ia];
    pc->t0[ia] = ref.t0[ia];
    pc->force[ia] = ref.force[ia];
    pc->torque[ia] = ref.torque[ia];
    pc->fc0[ia] = ref.fc0[ia];
    pc->tc0[ia] = ref.tc0[ia];
  }

  for (ia = 0; ia < 21; ia++) {
    pc->zeta[ia] = ref.zeta[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_assert
 *
 *  Assert that two colloids hold the same summed quantities.
 *
 *****************************************************************************/

static int test_colloid_sums_assert(colloid_t c1, colloid_t * c2) {

  int ia;

  assert(c2);

  /* STRUCTURE */

  test_assert(fabs(c1.sumw - c2->sumw) < TEST_DOUBLE_TOLERANCE);

  for (ia = 0; ia < 3; ia++) {
    test_assert(fabs(c1.cbar[ia] - c2->cbar[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.rxcbar[ia] - c2->rxcbar[ia]) < TEST_DOUBLE_TOLERANCE);
  }

  /* DYNAMICS */

  test_assert(fabs(c1.sump - c2->sump) < TEST_DOUBLE_TOLERANCE);

  for (ia = 0; ia < 3; ia++) {
    test_assert(fabs(c1.f0[ia] - c2->f0[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.t0[ia] - c2->t0[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.force[ia] - c2->force[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.torque[ia] - c2->torque[ia]) < TEST_DOUBLE_TOLERANCE);
  }

  for (ia = 0; ia < 21; ia++) {
    test_assert(fabs(c1.zeta[ia] - c2->zeta[ia]) < TEST_DOUBLE_TOLERANCE);
  }

  /* ACTIVE */

  for (ia = 0; ia < 3; ia++) {
    test_assert(fabs(c1.fc0[ia] - c2->fc0[ia]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(c1.tc0[ia] - c2->tc0[ia]) < TEST_DOUBLE_TOLERANCE);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_move
 *
 *  Here we move the particle across the lattice as a smoke test,
 *  so success is just getting to the end.
 *
 *****************************************************************************/

static int test_colloid_sums_move(pe_t * pe) {

  int index;
  int ic, jc, kc;
  int ntotal[3] = {64, 64, 64};
  int n, nstep = 100;
  int ncell[3] = {8, 8, 8};
  double r0[3] = {56.55, 8.55, 3.0};
  double dx;

  cs_t * cs = NULL;
  colloid_t * pc;
  colloids_info_t * cinfo = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  colloids_info_create(pe, cs, ncell, &cinfo);
  assert(cinfo);

  dx = 1.0*ntotal[X]/nstep;

  index = 1;
  colloids_info_add_local(cinfo, index, r0, &pc);

  colloids_halo_state(cinfo);
  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);

  for (n = 0; n < 2*nstep; n++) {

    /* Move the particle (twice around) */

    for (ic = 0; ic <= ncell[X] + 1; ic++) {
      for (jc = 0; jc <= ncell[Y] + 1; jc++) {
	for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	  colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	  while (pc) {
	    pc->s.r[Y] -= dx;
	    pc = pc->next;
	  }
	  
	}
      }
    }

    colloids_info_update_cell_list(cinfo);
    colloids_halo_state(cinfo);
    colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);
  }

  /* Success */

  colloids_info_free(cinfo);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_sums_conservation
 *
 *  Test conservation meesage. This is not a real test in that
 *  no particle is built and/or moved.
 *
 *****************************************************************************/

int test_colloid_sums_conservation(pe_t * pe) {

  int index;
  int ic, jc, kc;
  int ntotal[3] = {64, 64, 64};
  int ncell[3] = {8, 8, 8};
  double r0[3] = {32.0, 32.0, 32.0};

  cs_t * cs = NULL;
  colloid_t * pc;
  colloids_info_t * cinfo = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  colloids_info_create(pe, cs, ncell, &cinfo);
  assert(cinfo);

  index = 1;
  colloids_info_add_local(cinfo, index, r0, &pc);

  /* Swap the halo with zero information before setting the
   * test quantities locally. */

  colloids_halo_state(cinfo);

  if (pc) {
    pc->s.deltaphi = 1.0;
    pc->dq[0]  = 10.0;
    pc->dq[1]  = 100.0;    
  }

  /* Make the sum, and check all copies. */

  colloid_sums_halo(cinfo, COLLOID_SUM_CONSERVATION);

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  assert(fabs(pc->s.deltaphi - 1.0) < DBL_EPSILON);
	  assert(fabs(pc->dq[0]  - 10.0) < DBL_EPSILON);
	  assert(fabs(pc->dq[1]  - 100.0) < DBL_EPSILON);
	  pc = pc->next;
	}
	  
      }
    }
  }

  /* Success */

  colloids_info_free(cinfo);
  cs_free(cs);

  return 0;
}
