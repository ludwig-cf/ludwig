/*****************************************************************************
 *
 *  test_build.c
 *
 *  Test colloid build process, and integrity of links.
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "colloids_halo.h"
#include "colloid_sums.h"
#include "model.h"
#include "build.h"

int test_build_suite(void);
static int test_build_links_model_c1(double a0, double r0[3]);
static int test_build_links_model_c2(double a0, double r0[3]);
static int test_build_rebuild_c1(double a0, double r0[3]);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  test_build_suite();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_build_suite
 *
 *****************************************************************************/

int test_build_suite(void) {

  double a0;
  double r0[3];
  double delta = 1.0;        /* A small lattice offset */

  pe_init();
  coords_init();

  /* Distributions should not be required, but build_update() sets
   * 'internal velocity'; this should be split of as separate
   * concern. */

  distribution_init();

  a0 = 2.3;
  r0[X] = 0.5*L(X); r0[Y] = 0.5*L(Y); r0[Z] = 0.5*L(Z);
  test_build_links_model_c1(a0, r0);
  test_build_links_model_c2(a0, r0);
  test_build_rebuild_c1(a0, r0);

  a0 = 4.77;
  r0[X] = Lmin(X) + delta; r0[Y] = 0.5*L(Y); r0[Z] = 0.5*L(Z);
  test_build_links_model_c1(a0, r0);
  test_build_links_model_c2(a0, r0);
  test_build_rebuild_c1(a0, r0);

  a0 = 3.84;
  r0[X] = L(X); r0[Y] = L(Y); r0[Z] = L(Z);
  test_build_links_model_c1(a0, r0);
  test_build_links_model_c2(a0, r0);
  test_build_rebuild_c1(a0, r0);

  /* Some known cases: place the colloid in the centre and test only
   * in serial, as there is no quick way to compute in parallel. */

  distribution_finish();
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_build_links_model_c1
 *
 *  Plave a colloid, a0 = 2.3, in the centre of the system, and
 *  examine the link integrity.
 *
 *  We expect:
 *     \sum_b w_b c_b_alpha = 0         for all alpha;
 *     \sum_b w_b r_b x c_b_alpha = 0   ditto.
 *
 *  independent of the model.
 *
 *  Owner tests the result.
 *
 *****************************************************************************/

static int test_build_links_model_c1(double a0, double r0[3]) {

  int ncolloid;
  int ncell[3] = {2, 2, 2};

  map_t * map = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  colloids_info_create(ncell, &cinfo);
  colloids_info_map_init(cinfo);
  map_create(0, &map);

  /* Place the single colloid and construct the links */

  colloids_info_add_local(cinfo, 1, r0, &pc);
  if (pc) pc->s.a0 = a0;
  colloids_info_ntotal_set(cinfo);

  colloids_halo_state(cinfo);
  build_update_map(cinfo, map);
  build_update_links(cinfo, map);

  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);
  colloids_info_ntotal(cinfo, &ncolloid);
  assert(ncolloid == 1);

  /* The owner checks the details */

  if (pc) {

    /* DBL_EPSILON is too tight for these tests; FLT_EPSILON will catch
     * missing links which will be O(1). Could accumulate these sums as
     * integers to avoid this. */
    assert(fabs(pc->cbar[X] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->cbar[Y] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->cbar[Z] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->rxcbar[X] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->rxcbar[Y] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->rxcbar[Z] - 0.0) < FLT_EPSILON);
    assert(fabs(pc->deltam - 0.0) < DBL_EPSILON);
  }

  map_free(map);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 * As for _c1() above, but everyone checks the result.
 *
 *****************************************************************************/

static int test_build_links_model_c2(double a0, double r0[3]) {

  int ic, jc, kc;
  int ncolloid;
  int ncell[3] = {4, 4, 4};

  map_t * map = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  colloids_info_create(ncell, &cinfo);
  colloids_info_map_init(cinfo);
  map_create(0, &map);

  /* Place the single colloid and construct the links */

  colloids_info_add_local(cinfo, 1, r0, &pc);
  if (pc) pc->s.a0 = a0;
  colloids_info_ntotal_set(cinfo);

  colloids_halo_state(cinfo);
  build_update_map(cinfo, map);
  build_update_links(cinfo, map);

  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);
  colloids_info_ntotal(cinfo, &ncolloid);
  assert(ncolloid == 1);

  /* Everyone checks the details */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	if (pc) {

	  /* DBL_EPSILON is too tight for these tests; FLT_EPSILON will catch
	   * missing links which will be O(1). Could accumulate these sums as
	   * integers to avoid this. */
	  assert(fabs(pc->cbar[X] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->cbar[Y] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->cbar[Z] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[X] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[Y] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[Z] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->deltam - 0.0) < DBL_EPSILON);
	}

      }
    }
  }

  map_free(map);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_build_rebuild_c1
 *
 *****************************************************************************/

static int test_build_rebuild_c1(double a0, double r0[3]) {

  int ic, jc, kc;
  int ncolloid;
  int ncell[3] = {2, 2, 2};

  map_t * map = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  colloids_info_create(ncell, &cinfo);
  colloids_info_map_init(cinfo);
  map_create(0, &map);

  /* Place the single colloid and construct the links */

  colloids_info_add_local(cinfo, 1, r0, &pc);
  if (pc) {
    pc->s.a0 = a0;
    /* All copies must have this, applied later */
    pc->s.dr[X] = 0.0;
    pc->s.dr[Y] = 0.0;
    pc->s.dr[Z] = 0.0;
  }

  colloids_info_ntotal_set(cinfo);

  colloids_halo_state(cinfo);
  build_update_map(cinfo, map);
  build_update_links(cinfo, map);

  colloids_info_ntotal(cinfo, &ncolloid);
  assert(ncolloid == 1);

  /* Move, rebuild, and check */

  colloids_info_position_update(cinfo);
  colloids_info_update_cell_list(cinfo);
  colloids_halo_state(cinfo);

  build_update_map(cinfo, map);
  build_update_links(cinfo, map);
  colloid_sums_halo(cinfo, COLLOID_SUM_STRUCTURE);

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	if (pc) {

	  /* DBL_EPSILON is too tight for these tests; FLT_EPSILON will catch
	   * missing links which will be O(1). Could accumulate these sums as
	   * integers to avoid this. */
	  assert(fabs(pc->cbar[X] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->cbar[Y] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->cbar[Z] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[X] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[Y] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->rxcbar[Z] - 0.0) < FLT_EPSILON);
	  assert(fabs(pc->deltam - 0.0) < DBL_EPSILON);
	}

      }
    }
  }

  map_free(map);
  colloids_info_free(cinfo);

  return 0;
}
