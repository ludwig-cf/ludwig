/*****************************************************************************
 *
 *  test_build.c
 *
 *  Test colloid map.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "build.h"
#include "colloids_halo.h"
#include "util.h"

int test_build_update_map(pe_t * pe, cs_t * cs);

static int test_build_update_map_sph(pe_t * pe, cs_t * cs, double a0,
                                     const double r0[3]);
static int test_build_update_map_ell(pe_t * pe, cs_t * cs, const double abc[3],
                                     const double r0[3], const double q[4]);

/*****************************************************************************
 *
 *  test_build_suite
 *
 *****************************************************************************/

int test_build_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_build_update_map(pe, cs);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_build\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_build_update_map
 *
 *  At the moment build_update_map() updates a number of things:
 *    1. map status
 *    2. map data wetting values (not always relevant)
 *    3. colloid map
 *
 *  Those functions should probably be split; here we only examine
 *  the map status.
 *
 *****************************************************************************/

int test_build_update_map(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  {
    double a0    = 0.25;
    double r0[3] = {1.0, 1.0, 1.0}; /* Trivial case volume = 1 unit */

    ifail = test_build_update_map_sph(pe, cs, a0, r0);
    assert(ifail == 0);
  }

  {
    double a0    = 0.87;            /* A bit more than sqrt(3)/2 ... */
    double r0[3] = {1.5, 1.5, 1.5}; /* ... so have 8 sites inside */

    ifail = test_build_update_map_sph(pe, cs, a0, r0);
    assert(ifail == 0);
  }

  {
    /* prolate ellipsoid */
    double abc[3] = {1.01, 0.25, 0.25};
    double r0[3]  = {1.00, 1.00, 1.00};
    double q4[4]  = {1.00, 0.00, 0.00, 0.00};

    ifail = test_build_update_map_ell(pe, cs, abc, r0, q4);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_build_update_map_sph
 *
 *  Utility to test map against discrete volume for spheres.
 *
 *****************************************************************************/

static int test_build_update_map_sph(pe_t * pe, cs_t * cs, double a0,
                                     const double r0[3]) {

  int ifail    = 0;
  int ndata    = 2;
  int ncell[3] = {8, 8, 8};

  map_options_t opts = map_options_ndata(ndata);
  map_t *       map  = NULL;
  colloid_t *   pc   = NULL;

  colloid_options_t options = colloid_options_ncell(ncell);
  colloids_info_t * cinfo   = NULL;

  map_create(pe, cs, &opts, &map);

  colloids_info_create(pe, cs, &options, &cinfo);

  {
    double          ah = a0;
    colloid_state_t s  = {};
    colloid_state_init_sphere(1, a0, ah, r0, &s);
    colloids_info_add_local(cinfo, &s, &pc);
  }

  colloids_info_ntotal_set(cinfo);
  colloids_halo_state(cinfo);

  build_update_map(cinfo, map);

  {
    /* All ranks compute total and check */
    int    nvol = 0;
    double avol = 0.0;
    map_volume_allreduce(map, MAP_COLLOID, &nvol);
    util_discrete_volume_sphere(r0, a0, &avol);
    if (fabs(avol - 1.0 * nvol) > DBL_EPSILON) {
      ifail = -1;
    }
  }

  colloids_info_free(&cinfo);
  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_buld_update_map_ell
 *
 *  Utility to test map against discrete volume for ellipsoids
 *
 *****************************************************************************/

static int test_build_update_map_ell(pe_t * pe, cs_t * cs, const double abc[3],
                                     const double r0[3], const double q[4]) {
  int ifail    = 0;
  int ndata    = 2;
  int ncell[3] = {8, 8, 8};

  map_options_t opts = map_options_ndata(ndata);
  map_t *       map  = NULL;
  colloid_t *   pc   = NULL;

  colloid_options_t options = colloid_options_ncell(ncell);
  colloids_info_t * cinfo   = NULL;

  map_create(pe, cs, &opts, &map);

  colloids_info_create(pe, cs, &options, &cinfo);

  {
    colloid_state_t s = {};
    colloid_state_init_ellipsoid(1, abc, q, r0, &s);
    colloids_info_add_local(cinfo, &s, &pc);
  }

  colloids_info_ntotal_set(cinfo);
  colloids_halo_state(cinfo);

  build_update_map(cinfo, map);

  {
    /* All ranks compute total and check ... */
    int    nvol = 0;
    double avol = 0.0;
    map_volume_allreduce(map, MAP_COLLOID, &nvol);
    if (nvol != util_discrete_volume_ellipsoid(abc, r0, q, &avol)) {
      ifail = -1;
    }
  }

  colloids_info_free(&cinfo);
  map_free(&map);

  return ifail;
}
