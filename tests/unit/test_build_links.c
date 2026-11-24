/*****************************************************************************
 *
 *  test_build_links.c
 *
 *  Test integrity of links.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "build.h"
#include "build_links.h"
#include "colloids_halo.h"

int test_build_links_update_driver(pe_t * pe, cs_t * cs);

static int test_build_update_links_sph(pe_t * pe, cs_t * cs, double a0,
                                       const double r0[3], int nvel);

/*****************************************************************************
 *
 *  test_build_links_suite
 *
 *****************************************************************************/

int test_build_links_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_build_links_update_driver(pe, cs);

  cs_free(cs);
  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_build_links_update_driver
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

int test_build_links_update_driver(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  {
    double a0    = 0.25;
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
 *  test_build_update_links_sph
 *
 *  A utiity to check the number of links generated for the given model
 *  (nvel) is consistent. This is limited to COLLOID_SHAPE_SPHERE
 *  occupying a single site.
 *
 *****************************************************************************/

static int test_build_update_links_sph(pe_t * pe, cs_t * cs, double a0,
                                       const double r0[3], int nvel) {
  int ifail    = 0;
  int ndata    = 2;
  int ncell[3] = {8, 8, 8};

  map_options_t opts = map_options_ndata(ndata);
  map_t *       map  = NULL;

  lb_t *            lb         = NULL;
  lb_data_options_t lb_options = lb_data_options_default();

  colloid_options_t options = colloid_options_ncell(ncell);
  colloids_info_t * info    = NULL;

  options.nvel = nvel;
  colloids_info_create(pe, cs, &options, &info);

  map_create(pe, cs, &opts, &map);

  lb_options.nvel = nvel;
  lb_data_create(pe, cs, &lb_options, &lb);

  {
    double          ah = a0;
    colloid_t *     pc = NULL;
    colloid_state_t s  = {};

    colloid_state_init_sphere(1, a0, ah, r0, &s);
    colloids_info_add_local(info, &s, &pc);
  }

  colloids_info_ntotal_set(info);
  colloids_halo_state(info);
  colloids_info_update_lists(info);

  build_update_map(info, map);
  build_links_update_driver(info, NULL, map, lb);

  {
    /* Count up the number of links. Should be (nvel-1) globally */
    int      nlink = 0;
    MPI_Comm comm  = MPI_COMM_NULL;

    /* Remember to run through all halo images ... */
    for (int n = 0; n < info->npall; n++) {
      colloid_t * pc = info->pointers->colloid[n];
      nlink += pc->links->active_links;
    }

    /* All ranks check */
    pe_mpi_comm(pe, &comm);
    MPI_Allreduce(MPI_IN_PLACE, &nlink, 1, MPI_INT, MPI_SUM, comm);
    if (nlink != (nvel - 1)) {
      ifail = -1;
    }
  }

  lb_free(lb);
  map_free(&map);
  colloids_info_free(&info);

  return ifail;
}
