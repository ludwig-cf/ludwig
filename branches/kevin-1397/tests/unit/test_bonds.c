/*****************************************************************************
 *
 *  test_bonds.c
 *
 *  Test of bonded interactions, and associated angles.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "colloids_halo.h"
#include "colloids.h"
#include "tests.h"

int test_bonds_dimers(void);
int test_bonds_trimers(void);
int test_bonds_dimer_instance(double a0, double r1[3], double r2[3]);
int test_bonds_trimer_instance(double a0, double r1[3], double r2[3],
			       double r3[3]);
int colloid_forces_bonds_count_local(colloids_info_t * cinfo, int * nbond,
				     int * nangle);
int colloid_forces_bonds_check(colloids_info_t * cinfo, int * nbondfound,
			       int * nbondpair);

/*****************************************************************************
 *
 *  test_bonds_suite
 *
 *****************************************************************************/

int test_bonds_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  coords_init();

  test_bonds_dimers();
  test_bonds_trimers();

  coords_finish();
  pe_info(pe, "PASS     ./unit/test_bonds\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_bonds_dimers
 *
 *****************************************************************************/

int test_bonds_dimers(void) {

  double a0 = 2.3;  /* Colloid radius doubling as 'bond length' */
  double r1[3];
  double r2[3];

  r1[X] = 0.5*L(X) - a0;
  r1[Y] = 0.5*L(Y) - a0;
  r1[Z] = 0.5*L(Z) - a0;
  r2[X] = 0.5*L(X) + a0;
  r2[Y] = 0.5*L(X) + a0;
  r2[Z] = 0.5*L(X) + a0;

  test_bonds_dimer_instance(a0, r1, r2);
  test_bonds_dimer_instance(a0, r2, r1);

  r1[X] = a0;
  r1[Y] = a0;
  r1[Z] = a0;
  r2[X] = L(X) - a0;
  r2[Y] = L(Y) - a0;
  r2[Z] = L(Z) - a0;

  test_bonds_dimer_instance(a0, r1, r2);
  test_bonds_dimer_instance(a0, r2, r1);

  return 0;
}

/*****************************************************************************
 *
 *  test_bonds_trimers
 *
 *****************************************************************************/

int test_bonds_trimers(void) {

  double a0 = 2.3;
  double r0[3];
  double r1[3];
  double r2[3];

  /* Straight trimer */

  r0[X] = 0.5*L(X) - a0;
  r0[Y] = 0.5*L(Y);
  r0[Z] = 0.5*L(Z);

  r1[X] = r0[X] + 3.0*a0;
  r1[Y] = r0[Y];
  r1[Z] = r0[Z];

  r2[X] = r0[X] - 3.0*a0;
  r2[Y] = r0[Y];
  r2[Z] = r0[Z];

  test_bonds_trimer_instance(a0, r0, r1, r2);
  test_bonds_trimer_instance(a0, r0, r2, r1);

  /* L-shape trimer */

  r1[X] = L(X) - a0;
  r1[Y] = L(Y) - a0;
  r1[Z] = L(Z) - a0;

  r0[X] = a0;
  r0[Y] = r1[Y];
  r0[Z] = r1[Z];

  r2[X] = a0;
  r2[Y] = a0;
  r2[Z] = r1[Z];

  test_bonds_trimer_instance(a0, r0, r1, r2);
  test_bonds_trimer_instance(a0, r0, r2, r1);

  r2[Z] = a0;

  test_bonds_trimer_instance(a0, r0, r1, r2);
  test_bonds_trimer_instance(a0, r0, r2, r1);

  /* Small angle */

  r1[X] = 0.4*L(X);
  r1[Y] = 0.4*L(Y);
  r1[Z] = 0.4*L(Z);

  r0[X] = r1[X] - 1.5*a0;
  r0[Y] = r1[Y] - 0.1;
  r0[Z] = r1[Z];

  r2[X] = r0[X] + 1.5*a0;
  r2[Y] = r0[Y] - 0.1;
  r2[Z] = r0[Z];

  test_bonds_trimer_instance(a0, r0, r1, r2);

  return 0;
}

/*****************************************************************************
 *
 *  test_bonds_dimer_instance
 *
 *  Two colloids sharing one bond. No angles.
 *  Place two particles, radius a0, at r1 and r2
 *  Assume a0 < 0.5 |r12|, although it doesn't really matter here.
 *
 *****************************************************************************/

int test_bonds_dimer_instance(double a0, double r1[3], double r2[3]) {

  int nc;
  int nbond, nbond_local;
  int nangle, nangle_local;
  int npair, npair_local;
  int ncell[3] = {2, 2, 2};
  colloids_info_t * cinfo = NULL;

  colloid_t * pc = NULL;
  colloid_state_t state1;
  colloid_state_t state2;
  colloid_state_t * state0;

  MPI_Comm comm = cart_comm();

  colloids_info_create(ncell, &cinfo);
  assert(cinfo);

  state0 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
  assert(state0);

  state1 = *state0;

  state1.index = 1;
  state1.r[X] = r1[X];
  state1.r[Y] = r1[Y];
  state1.r[Z] = r1[Z];

  colloids_info_add_local(cinfo, state1.index, state1.r, &pc);
  if (pc) {
    pc->s.a0 = a0;
    pc->s.ah = a0;
    pc->s.nbonds = 1;
    pc->s.bond[0] = 2;
  }

  /* TWO */

  state2 = *state0;
  state2.index = 2;
  state2.r[X] = r2[X];
  state2.r[Y] = r2[Y];
  state2.r[Z] = r2[Z];

  pc = NULL;
  colloids_info_add_local(cinfo, state2.index, state2.r, &pc);
  if (pc) {
    pc->s.a0 = a0;
    pc->s.ah = a0;
    pc->s.nbonds = 1;
    pc->s.bond[0] = 1;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);
  colloids_info_list_local_build(cinfo);

  colloid_forces_bonds_count_local(cinfo, &nbond_local, &nangle_local);

  MPI_Allreduce(&nbond_local, &nbond, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&nangle_local, &nangle, 1, MPI_INT, MPI_SUM, comm);

  assert(nbond == 1);
  assert(nangle == 0);

  colloid_forces_bonds_check(cinfo, &nbond_local, &npair_local);

  MPI_Allreduce(&nbond_local, &nbond, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&npair_local, &npair, 1, MPI_INT, MPI_SUM, comm);

  assert(nbond == 1);
  assert(npair == 1);

  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_bonds_trimer_instance
 *
 *  Two bonds plus one angle. r0 is the position of the 'central'
 *  colloid.
 *
 *****************************************************************************/

int test_bonds_trimer_instance(double a0, double r0[3], double r1[3],
			       double r2[3]) {

  int nc;
  int nbond, nbond_local;
  int nangle, nangle_local;
  int npair, npair_local;

  int ncell[3] = {2, 2, 2};
  colloids_info_t * cinfo = NULL;

  colloid_t * pc = NULL;
  colloid_state_t state0;
  colloid_state_t state1;
  colloid_state_t state2;
  colloid_state_t * state_null;

  MPI_Comm comm = cart_comm();

  colloids_info_create(ncell, &cinfo);
  assert(cinfo);

  state_null = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
  assert(state_null);

  /* Central particle: two bonds plus one angle */

  state0 = *state_null;

  state0.index = 1;
  state0.r[X] = r0[X];
  state0.r[Y] = r0[Y];
  state0.r[Z] = r0[Z];

  colloids_info_add_local(cinfo, state0.index, state0.r, &pc);
  if (pc) {
    pc->s.a0 = a0;
    pc->s.ah = a0;
    pc->s.nbonds = 2;
    pc->s.bond[0] = 2;
    pc->s.bond[1] = 3;
    pc->s.nangles = 1;
  }

  /* Two */

  state1 = *state_null;

  state1.index = 2;
  state1.r[X] = r1[X];
  state1.r[Y] = r1[Y];
  state1.r[Z] = r1[Z];

  pc = NULL;
  colloids_info_add_local(cinfo, state1.index, state1.r, &pc);
  if (pc) {
    pc->s.a0 = a0;
    pc->s.ah = a0;
    pc->s.nbonds = 1;
    pc->s.bond[0] = 1;
  }

  /* Three */

  state2 = *state_null;

  state2.index = 3;
  state2.r[X] = r2[X];
  state2.r[Y] = r2[Y];
  state2.r[Z] = r2[Z];

  pc = NULL;
  colloids_info_add_local(cinfo, state2.index, state2.r, &pc);
  if (pc) {
    pc->s.a0 = a0;
    pc->s.ah = a0;
    pc->s.nbonds = 1;
    pc->s.bond[0] = 1;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 3);

  colloids_halo_state(cinfo);
  colloids_info_list_local_build(cinfo);

  colloid_forces_bonds_count_local(cinfo, &nbond_local, &nangle_local);

  MPI_Allreduce(&nbond_local, &nbond, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&nangle_local, &nangle, 1, MPI_INT, MPI_SUM, comm);

  assert(nbond == 2);
  assert(nangle == 1);

  colloid_forces_bonds_check(cinfo, &nbond_local, &npair_local);

  MPI_Allreduce(&nbond_local, &nbond, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&npair_local, &npair, 1, MPI_INT, MPI_SUM, comm);

  assert(nbond == 2);
  assert(npair == 2);

  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_bonds_count_local
 *
 *  Count bonds and angles for local colloids.
 *
 *****************************************************************************/

int colloid_forces_bonds_count_local(colloids_info_t * cinfo, int * nbond,
				     int * nangle) {
  int n;
  colloid_t * pc = NULL;

  assert(cinfo);
  assert(nbond);
  assert(nangle);

  *nbond = 0;
  *nangle = 0;

  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {
    for (n = 0; n < pc->s.nbonds; n++) {
      /* Single-count bonds */
      if (pc->s.index < pc->s.bond[n]) *nbond += 1;
    }
    *nangle += pc->s.nangles; 
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_bonds_check
 *
 *  If bonds are present, check that a bond A -> B implies that
 *  the corresponding B -> A is visible.
 *
 *****************************************************************************/

int colloid_forces_bonds_check(colloids_info_t * cinfo, int * nbondfound,
			       int * nbondpair) {

  int ic, jc, kc;
  int ncell[3];
  int n1, n2;

  int id, jd, kd;
  int ic1, ic2, jc1, jc2, kc1, kc2;
  int range[3];
  int halo [3];

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  *nbondfound = 0;
  *nbondpair  = 0;

  range[X] = 1 + (ncell[X] == 2);
  range[Y] = 1 + (ncell[Y] == 2);
  range[Z] = 1 + (ncell[Z] == 2);
  halo[X] = (cart_size(X) > 1 || range[X] == 1);
  halo[Y] = (cart_size(Y) > 1 || range[Y] == 1);
  halo[Z] = (cart_size(Z) > 1 || range[Z] == 1);

  for (ic = 1; ic <= ncell[X]; ic++) {
    ic1 = imax(1 - halo[X], ic - range[X]);
    ic2 = imin(ic + range[X], ncell[X] + halo[X]);

    for (jc = 1; jc <= ncell[Y]; jc++) {
      jc1 = imax(1 - halo[Y], jc - range[Y]);
      jc2 = imin(jc + range[Y], ncell[Y] + halo[Y]);

      for (kc = 1; kc <= ncell[Z]; kc++) {
        kc1 = imax(1 - halo[Z], kc - range[Z]);
        kc2 = imin(kc + range[Z], ncell[Z] + halo[Z]);

	for (id = ic1; id <= ic2; id++) {
	  for (jd = jc1; jd <= jc2; jd++) {
	    for (kd = kc1; kd <= kc2; kd++) {

	      colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc1);

	      for (; pc1; pc1 = pc1->next) {

		colloids_info_cell_list_head(cinfo, id, jd, kd, &pc2);

		for (; pc2; pc2 = pc2->next) {

		  /* Do not double-count condition (i < j) */
		  if (pc1->s.index >= pc2->s.index) continue;

		  /* For each bond pc1 ... is it pc2? */

		  for (n1 = 0; n1 < pc1->s.nbonds; n1++) {
		    if (pc1->s.bond[n1] == pc2->s.index) {
		      *nbondfound += 1;
		      pc1->bonded[n1] = pc2;
		      /* And bond is reciprocated */
		      for (n2 = 0; n2 < pc2->s.nbonds; n2++) {
			if (pc2->s.bond[n2] == pc1->s.index) {
			  *nbondpair += 1;
			  pc2->bonded[n2] = pc1;
			}
		      }
		    }
		  }
		}
	      }

	      /* Inner cell list */
	    }
	  }
	}

	/* Outer cell list. */
      }
    }
  }

  return 0;
}
