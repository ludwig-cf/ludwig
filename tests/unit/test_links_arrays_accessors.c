#include <assert.h>

#include "colloids.h"
#include "colloid_link.h"
#include "tests.h"

/*****************************************************************************
 *
 *  test_links_arrays_accessors
 *
 *****************************************************************************/

int test_links_arrays_accessors(void) {
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
  colloids_info_add_local(cinfo, 1, r, 1.0, &pc);

  pc->lnk->i = 3;
  pc->lnk->j = 2;
  pc->lnk->p = 1;
  pc->lnk->status = 0;
  for (int i = 0; i < 3; i++)
    pc->lnk->rb[i] = i;

  copy_links_to_array(pc);

  int i, j, p, status;
  double rb[3];

  colloid_link_i(pc->links, 0, &i);
  colloid_link_j(pc->links, 0, &j);
  colloid_link_p(pc->links, 0, &p);
  colloid_link_status(pc->links, 0, &status);
  colloid_link_rb(pc->links, 0, rb);

  assert(i == pc->lnk->i);
  assert(j == pc->lnk->j);
  assert(p == pc->lnk->p);
  assert(status == pc->lnk->status);
  for (int index = 0; index < 3; index++)
    assert(rb[index] == pc->lnk->rb[index]);

  return 0;
}