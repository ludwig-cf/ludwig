/*****************************************************************************
 *
 *  test_wall_ss_cut.c
 *
 *  Test for wall-colloid 'pair' interaction.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2024 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "wall_ss_cut.h"

int test_wall_ss_cut_create(pe_t * pe, cs_t * cs, wall_t * wall);
int test_wall_ss_cut_single(pe_t * pe, cs_t * cs, wall_t * wall);
int test_wall_ss_cut_compute(pe_t * pe, cs_t * cs, wall_t * wall);

/*****************************************************************************
 *
 *  test_wall_ss_cut_suite
 *
 *****************************************************************************/

int test_wall_ss_cut_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  {
    lb_t * lb = NULL;
    map_t * map = NULL;
    wall_t * wall = NULL;

    wall_param_t param = {.iswall = 1, .isboundary = {1,1,1}};
    map_options_t mapopts = map_options_default();
    lb_data_options_t opts = lb_data_options_default();

    lb_data_create(pe, cs, &opts, &lb);
    map_create(pe, cs, &mapopts, &map);
    wall_create(pe, cs, map, lb, &wall);
    wall_commit(wall, &param);

    test_wall_ss_cut_create(pe, cs, wall);
    test_wall_ss_cut_single(pe, cs, wall);
    test_wall_ss_cut_compute(pe, cs, wall);

    wall_free(wall);
    map_free(&map);
    lb_free(lb);
  }

  pe_info(pe, "PASS     ./unit/test_wall_ss_cut\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_ss_cut_create
 *
 *****************************************************************************/

int test_wall_ss_cut_create(pe_t * pe, cs_t * cs, wall_t * wall) {

  wall_ss_cut_t * wall_ss_cut = NULL;
  wall_ss_cut_options_t opts = {.epsilon = 0.1, .sigma = 0.2, .nu = 0.3,
                                .hc = 0.4};
  assert(pe);
  assert(cs);
  assert(wall);

  wall_ss_cut_create(pe, cs, wall, &opts, &wall_ss_cut);
  assert(wall_ss_cut);

  wall_ss_cut_free(wall_ss_cut);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_ss_cut_single
 *
 *****************************************************************************/

int test_wall_ss_cut_single(pe_t * pe, cs_t * cs, wall_t * wall) {

  wall_ss_cut_t * wall_ss_cut = NULL;
  wall_ss_cut_options_t opts = {.epsilon = 0.001,
                                .sigma = 0.8,
				.nu = 2.0,
				.hc = 0.25};
  assert(pe);
  assert(cs);
  assert(wall);

  wall_ss_cut_create(pe, cs, wall, &opts, &wall_ss_cut);

  {
    double h = 0.0125;
    double f = 0.0;
    double v = 0.0;

    wall_ss_cut_single(wall_ss_cut, h, &f, &v);
    assert(fabs(f - 655.27808) < FLT_EPSILON);
    assert(fabs(v - 4.0663040) < FLT_EPSILON);
  }
  wall_ss_cut_free(wall_ss_cut);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_ss_cut_compute
 *
 *****************************************************************************/

int test_wall_ss_cut_compute(pe_t * pe, cs_t * cs, wall_t * wall) {

  int ncell[3] = {2, 2, 2};
  colloids_info_t * cinfo = NULL;

  wall_ss_cut_t * wall_ss_cut = NULL;
  wall_ss_cut_options_t opts = {.epsilon = 0.001,
                                .sigma = 0.8,
				.nu = 2.0,
				.hc = 0.25};

  assert(pe);
  assert(cs);
  assert(wall);

  colloids_info_create(pe, cs, ncell, &cinfo);
  wall_ss_cut_create(pe, cs, wall, &opts, &wall_ss_cut);

  {
    /* Add a colloid at a suitable position */
    double a0 = 1.0;
    double ah = 1.0;
    double h  = 0.0125;
    double r[3] = {0.5 + a0 + h, 0.5 + a0 + opts.hc, 0.5 + a0 + opts.hc};
    colloid_t * pc = NULL;

    colloids_info_add_local(cinfo, 1, r, &pc);
    if (pc) {
      pc->s.a0 = a0;
      pc->s.ah = ah;
    }
    /* Need the local list up-to-date... */
    colloids_info_list_local_build(cinfo);

    wall_ss_cut_compute(cinfo, wall_ss_cut);
    if (pc) {
      assert(fabs(pc->force[X] - 655.27808) < FLT_EPSILON);
      assert(fabs(pc->force[Y] - 0.0)       < DBL_EPSILON);
      assert(fabs(pc->force[Z] - 0.0)       < DBL_EPSILON);
    }
  }

  wall_ss_cut_free(wall_ss_cut);
  colloids_info_free(cinfo);

  return 0;
}
