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
    if (pc == NULL) return; // Don't fail if there are no local colloids
    test_assert(pc->links != NULL);
    test_assert(pc->links->max_links == nlinks);
    
    pc->links->i[0] = 1;
    pc->links->j[0] = 1;
    pc->links->p[0] = 1;
    pc->links->status[0] = 1;
    for (int j = 0; j < 3; j++) 
      pc->links->rb[0][j] = 1;
    
    pc->links->i[nlinks-1] = 1;
    pc->links->j[nlinks-1] = 1;
    pc->links->p[nlinks-1] = 1;
    pc->links->status[nlinks-1] = 1;
    for (int j = 0; j < 3; j++) 
      pc->links->rb[nlinks-1][j] = 1;
 }

/*****************************************************************************
 *
 *  test_links_array
 *
 *****************************************************************************/

int test_links_array(double a0) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  colloids_info_t * cinfo = NULL;
  colloid_options_t opts = colloid_options_default();
  colloid_t * pc;
  double r[3] = {0.0, 0.0, 0.0};

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, &opts, &cinfo);
  colloids_info_add_local(cinfo, 1, r, a0, &pc);

  int nlinks = colloid_link_max_3d(a0, cinfo->options.nvel);
  
  test_links_allocated(pc, nlinks);

  colloid_options_t new_opts = cinfo->options;
  colloids_info_recreate(&new_opts, &cinfo);
  colloids_info_add_local(cinfo, 1, r, a0, &pc);
  test_links_allocated(pc, nlinks);

  return 0;
}

/*****************************************************************************
 *
 *  test_create_links_arrays_suite
 *
 *****************************************************************************/

 int test_create_links_arrays_suite(void) {
   test_links_array(0.0);
   test_links_array(2.3);

   return 0;
 }