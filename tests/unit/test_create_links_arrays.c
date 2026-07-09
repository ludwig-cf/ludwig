#include <assert.h>

#include "colloids.h"
#include "colloid_link.h"
#include "tests.h"

/*****************************************************************************
 *
 *  test_links_allocated
 *
 *****************************************************************************/

 void test_links_allocated(colloid_t * pc, int nlinks) {
    test_assert(pc->links != NULL);
    test_assert(pc->links->max_links == nlinks);
    
    pc->links->i[0] = 1;
    pc->links->j[0] = 1;
    pc->links->p[0] = 1;
    pc->links->status[0] = 1;
    for (int j = 0; j < 3; j++) 
      pc->links->rb[j][0] = 1;
    
    pc->links->i[nlinks-1] = 1;
    pc->links->j[nlinks-1] = 1;
    pc->links->p[nlinks-1] = 1;
    pc->links->status[nlinks-1] = 1;
    for (int j = 0; j < 3; j++) 
      pc->links->rb[j][nlinks-1] = 1;
 }

/*****************************************************************************
 *
 *  test_links_array
 *
 *****************************************************************************/

int test_links_array(pe_t *pe, double a0) {

  cs_t * cs = NULL;
  colloids_info_t * cinfo = NULL;
  colloid_options_t opts = colloid_options_default();
  colloid_t * pc;
  double r[3] = {0.5, 0.5, 0.5};

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, &opts, &cinfo);
  colloids_info_add_local(cinfo, 1, r, a0, &pc);

  // We've only initialised one colloid, so if a rank doesn't have a colloid don't do the test.
  if (pc) {
    int nlinks = colloid_link_max_3d(a0, cinfo->options.nvel);
  
    test_links_allocated(pc, nlinks);

    colloid_free_links_arrays(pc);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_links_array_with_state
 *
 *****************************************************************************/

int test_links_array_with_state(pe_t *pe, double a0) {

  cs_t * cs = NULL;
  colloids_info_t * cinfo = NULL;
  colloid_options_t opts = colloid_options_default();
  colloid_t * pc;
  colloid_state_t state;
  double r[3] = {0.5, 0.5, 0.5};

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, &opts, &cinfo);
  create_dummy_state(&state, 1, a0, r);
  colloids_info_add_local_with_state(cinfo, &state, &pc);

  // We've only initialised one colloid, so if a rank doesn't have a colloid don't do the test.
  if (pc) {
    int nlinks = colloid_link_max_3d(a0, cinfo->options.nvel);
  
    test_links_allocated(pc, nlinks);

    colloid_free_links_arrays(pc);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_create_links_arrays_suite
 *
 *****************************************************************************/

 int test_create_links_arrays_suite(void) {
   pe_t * pe = NULL;
   pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

   test_links_array(pe, 0.0);
   test_links_array(pe, 2.3);
   test_links_array_with_state(pe, 0.0);
   test_links_array_with_state(pe, 2.3);
   MPI_Barrier(MPI_COMM_WORLD);
   pe_info(pe, "PASS     ./unit/test_create_links_array\n");

   return 0;
 }