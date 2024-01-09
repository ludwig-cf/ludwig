/*****************************************************************************
 *
 *  test_build.c
 *
 *  Test colloid build process, and integrity of links.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "build.h"
#include "tests.h"
#include "util.h"
#include "util_ellipsoid.h"

int test_build_update_map(pe_t * pe, cs_t * cs);
int test_build_update_links(pe_t * pe, cs_t * cs);

static int test_build_update_map_sph(pe_t * pe, cs_t * cs, double a0,
				     const double r0[3]);
static int test_build_update_map_ell(pe_t * pe, cs_t * cs, const double abc[3],
				     const double r0[3], const double q[4]);
static int test_build_update_links_sph(pe_t * pe, cs_t * cs, double a0,
				       const double r0[3], int nvel);

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
  test_build_update_links(pe, cs);

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
    double a0 = 0.25;
    double r0[3] = {1.0, 1.0, 1.0};   /* Trivial case volume = 1 unit */

    ifail = test_build_update_map_sph(pe, cs, a0, r0);
    assert(ifail == 0);
  }

  {
    double a0 = 0.87;                 /* A bit more than sqrt(3)/2 ... */
    double r0[3] = {1.5, 1.5, 1.5};   /* ... so have 8 sites inside */

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
 *  test_build_update_links
 *
 *  Again, build_update_links() performs a number of different operations.
 *  Here we are concerned with just the generation of a consistent
 *  number of links (really not much more than a smoke test).
 *
 *  In contrast to build_update_map(), this also involves the lb_t model.
 *  There is a very simple check on the number of links for an object
 *  occupying one site, ie., the number of links is nvel.
 *
 *****************************************************************************/

int test_build_update_links(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  {
    double a0 = 0.25;
    double r0[3] = {1.0, 1.0, 1.0};

    ifail = test_build_update_links_sph(pe, cs, a0, r0, 15);
    assert(ifail == 0);
    ifail = test_build_update_links_sph(pe, cs, a0, r0, 19);
    assert(ifail == 0);
    ifail = test_build_update_links_sph(pe, cs, a0, r0, 27);
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

  int ifail = 0;
  int ncell[3] = {8, 8, 8};

  map_t * map = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  map_create(pe, cs, 0, &map);

  colloids_info_create(pe, cs, ncell, &cinfo);
  colloids_info_map_init(cinfo);

  {
    colloid_state_t s = {
      .index = 1,
      .rebuild = 1,
      .bc = COLLOID_BC_BBL,
      .shape = COLLOID_SHAPE_SPHERE,
      .a0 = a0,
      .r = {r0[X], r0[Y], r0[Z]}
    };
    colloids_info_add_local(cinfo, 1, r0, &pc);
    if (pc) pc->s = s;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_halo_state(cinfo);

  build_update_map(cs, cinfo, map);

  {
    /* All ranks compute total and check */
    int    nvol = 0;
    double avol = 0.0;
    map_volume_allreduce(map, MAP_COLLOID, &nvol);
    util_discrete_volume_sphere(r0, a0, &avol);
    if (fabs(avol - 1.0*nvol) > DBL_EPSILON) ifail = -1;
  }

  colloids_info_free(cinfo);
  map_free(map);

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
  int ifail = 0;
  int ncell[3] = {8, 8, 8};

  map_t * map = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  map_create(pe, cs, 0, &map);

  colloids_info_create(pe, cs, ncell, &cinfo);
  colloids_info_map_init(cinfo);

  {
    colloid_state_t s = {
      .index = 1,
      .rebuild = 1,
      .bc = COLLOID_BC_BBL,
      .shape = COLLOID_SHAPE_ELLIPSOID,
      .r = {r0[X], r0[Y], r0[Z]},
      .elabc = {abc[X], abc[Y], abc[Z]},
      .quat = {q[0], q[1], q[2], q[3]}
    };
    colloids_info_add_local(cinfo, 1, r0, &pc);
    if (pc) pc->s = s;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_halo_state(cinfo);

  build_update_map(cs, cinfo, map);

  {
    /* All ranks compute total and check ... */
    int    nvol = 0;
    double avol = 0.0;
    map_volume_allreduce(map, MAP_COLLOID, &nvol);
    if (nvol != util_discrete_volume_ellipsoid(abc, r0, q, &avol)) {
      ifail = -1;
    }
  }

  colloids_info_free(cinfo);
  map_free(map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_build_update_links_sph
 *
 *  A utiity to check the number of links generated for the given model
 *  (nvel) is consistent. This is limited to COLLOID_SHAPE_SPHERE
 *  occupying a single site.
 *
 *****************************************************************************/

static int test_build_update_links_sph(pe_t * pe, cs_t * cs, double a0,
				       const double r0[3], int nvel) {
  int ifail = 0;
  int ncell[3] = {8, 8, 8};

  map_t * map = NULL;
  lb_model_t lb = {0};
  colloids_info_t * cinfo = NULL;

  colloids_info_create(pe, cs, ncell, &cinfo);
  colloids_info_map_init(cinfo);

  map_create(pe, cs, 0, &map);
  lb_model_create(nvel, &lb);

  {
    colloid_t * pc = NULL;
    colloid_state_t s = {
      .index = 1,
      .rebuild = 1,
      .bc = COLLOID_BC_BBL,
      .shape = COLLOID_SHAPE_SPHERE,
      .a0 = a0,
      .r = {r0[X], r0[Y], r0[Z]}
    };
    colloids_info_add_local(cinfo, 1, r0, &pc);
    if (pc) pc->s = s;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_halo_state(cinfo);
  colloids_info_update_lists(cinfo);

  build_update_map(cs, cinfo, map);
  build_update_links(cs, cinfo, NULL, map, &lb);

  {
    /* Count up the number of links. Should be (nvel-1) globally */
    int nlink = 0;
    MPI_Comm comm = MPI_COMM_NULL;
    colloid_t * pc = NULL;

    /* Remmeber to run through all halo images ... */
    colloids_info_all_head(cinfo, &pc);
    for (; pc; pc = pc->nextall) {
      colloid_link_t * link = pc->lnk;
      for ( ; link; link = link->next) nlink += 1;
    }

    /* All ranks check */
    pe_mpi_comm(pe, &comm);
    MPI_Allreduce(MPI_IN_PLACE, &nlink, 1, MPI_INT, MPI_SUM, comm);
    if (nlink != (nvel - 1)) ifail = -1;
  }

  lb_model_free(&lb);
  map_free(map);
  colloids_info_free(cinfo);

  return ifail;
}
