/*****************************************************************************
 *
 *  test_colloids_halo.c
 *
 *  Halo swap test.
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
#include "tests.h"

static void test_colloids_halo111(void);
static void test_colloids_halo211(void);
static void test_colloids_halo_repeat(void);
static void test_position(const double * r1, const double * r2);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3] = {1024, 1024, 1024};

  pe_init(argc, argv);
  coords_ntotal_set(ntotal);
  coords_init();

  info("Colloid state halo swap test\n");

  colloids_init();
  test_colloids_halo111();
  colloids_finish();

  colloids_init();
  test_colloids_halo211();
  colloids_finish();

  colloids_init();
  test_colloids_halo_repeat();
  colloids_finish();

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_halo111
 *
 *  Please one colloid in the bottom left corner locally and swap.
 *  So the local cell is {1, 1, 1}.
 *
 *****************************************************************************/

void test_colloids_halo111(void) {

  int noffset[3];
  int ncount[2];
  int index;
  double r0[3];
  double r1[3];

  colloid_t * pc;

  coords_nlocal_offset(noffset);

  r0[X] = Lmin(X) + 1.0*noffset[X];
  r0[Y] = Lmin(Y) + 1.0*(noffset[Y] + 1);
  r0[Z] = Lmin(Z) + 1.0*(noffset[Z] + 1);

  index = 1 + pe_rank();
  colloid_add_local(index, r0);

  colloids_halo_send_count(X, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 1);

  colloids_halo_send_count(Y, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 1);

  colloids_halo_send_count(Z, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 1);

  colloids_halo_dim(X);

  /* All process should now have one particle in upper x halo region,
   * and the send count in Y (back) should be 2 */

  test_assert(colloids_cell_count(Ncell(X)+1, 1, 1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 2);

  pc = colloids_cell_list(Ncell(X)+1, 1, 1);
  test_assert(pc != NULL);
  r1[X] = r0[X] + 1.0*N_total(X)/cart_size(X);
  r1[Y] = r0[Y];
  r1[Z] = r0[Z];

  test_position(r1, pc->s.r);

  colloids_halo_send_count(Y, ncount);

  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 2);

  colloids_halo_dim(Y);

  /* There should now be two additional images, and the send count
   * goes up to 4 */

  test_assert(colloids_cell_count(1, Ncell(Y)+1, 1) == 1);
  test_assert(colloids_cell_count(Ncell(X)+1, Ncell(Y)+1, 1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 4);

  pc = colloids_cell_list(1, Ncell(Y)+1, 1);
  test_assert(pc != NULL);
  r1[X] = r0[X];
  r1[Y] = r0[Y] + 1.0*N_total(Y)/cart_size(Y);
  r1[Z] = r0[Z];
  test_position(r1, pc->s.r);

  colloids_halo_send_count(Z, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 4);

  colloids_halo_dim(Z);

  /* We should end up with eight in total, seven of which are
   * periodic images. */

  test_assert(colloids_cell_count(1, 1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(Ncell(X)+1, 1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(1, Ncell(Y)+1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(Ncell(X)+1, Ncell(Y)+1, Ncell(Z)+1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 8);

  return;
}

/*****************************************************************************
 *
 *  test_colloids_halo211
 *
 *  The local cell of the real particle is {2, 1, 1}.
 *
 *****************************************************************************/

void test_colloids_halo211(void) {

  int noffset[3];
  int ncount[2];
  int index;
  double r0[3];
  double r1[3];

  colloid_t * pc;

  coords_nlocal_offset(noffset);

  r0[X] = Lmin(X) + 1.0*noffset[X] + colloids_lcell(X);
  r0[Y] = Lmin(Y) + 1.0*(noffset[Y] + 1);
  r0[Z] = Lmin(Z) + 1.0*(noffset[Z] + 1);

  index = 1 + pe_rank();
  colloid_add_local(index, r0);
 
  colloids_halo_send_count(X, ncount);
  test_assert(ncount[FORWARD] == 1);
  test_assert(ncount[BACKWARD] == 0);

  colloids_halo_send_count(Y, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 1);

  colloids_halo_send_count(Z, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 1);

  colloids_halo_dim(X);

  /* All process should now have one particle in lower x halo region,
   * and the send count in Y (back) should be 2 */

  test_assert(colloids_cell_count(0, 1, 1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 2);

  pc = colloids_cell_list(0, 1, 1);
  test_assert(pc != NULL);
  r1[X] = r0[X] - 1.0*N_total(X)/cart_size(X);
  r1[Y] = r0[Y];
  r1[Z] = r0[Z];
  test_position(r1, pc->s.r);

  colloids_halo_send_count(Y, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 2);

  colloids_halo_dim(Y);

  /* There should now be two additional images, and the send count
   * goes up to 4 */

  test_assert(colloids_cell_count(2, Ncell(Y)+1, 1) == 1);
  test_assert(colloids_cell_count(0, Ncell(Y)+1, 1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 4);

  pc = colloids_cell_list(2, Ncell(Y)+1, 1);
  test_assert(pc != NULL);
  r1[X] = r0[X];
  r1[Y] = r0[Y] + 1.0*N_total(Y)/cart_size(Y);
  r1[Z] = r0[Z];
  test_position(r1, pc->s.r);

  colloids_halo_send_count(Z, ncount);
  test_assert(ncount[FORWARD] == 0);
  test_assert(ncount[BACKWARD] == 4);

  colloids_halo_dim(Z);

  /* We should end up with eight in total (locally) */

  test_assert(colloids_cell_count(2, 1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(0, 1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(2, Ncell(Y)+1, Ncell(Z)+1) == 1);
  test_assert(colloids_cell_count(0, Ncell(Y)+1, Ncell(Z)+1) == 1);
  test_assert(colloid_nlocal() == 1);
  test_assert(colloids_nalloc() == 8);

  return;
}

/*****************************************************************************
 *
 *  test_colloids_halo_repeat
 *
 *  Make sure repeat halo swap doesn't multiply particles.
 *
 *****************************************************************************/

void test_colloids_halo_repeat(void) {

  int noffset[3];
  int index;
  double r0[3];

  coords_nlocal_offset(noffset);

  r0[X] = Lmin(X) + 1.0*noffset[X];
  r0[Y] = Lmin(Y) + 1.0*(noffset[Y] + 1);
  r0[Z] = Lmin(Z) + 1.0*(noffset[Z] + 1);

  index = 1 + pe_rank();
  colloid_add_local(index, r0);
  index = 1 + pe_size() + pe_rank();
  colloid_add_local(index, r0);
  index = 1 + 2*pe_size() + pe_rank();
  colloid_add_local(index, r0);

  colloids_halo_state();
  colloids_halo_state();

  test_assert(colloid_nlocal() == 3);
  test_assert(colloids_nalloc() == 24);

  return;
}

/*****************************************************************************
 *
 *  test_position
 *
 *****************************************************************************/

void test_position(const double r1[3], const double r2[3]) {

  test_assert(fabs(r1[X] - r2[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r1[Y] - r2[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(r1[Z] - r2[Z]) < TEST_DOUBLE_TOLERANCE);

  return;
}
