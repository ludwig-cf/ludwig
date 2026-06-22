#include <assert.h>

#include "colloids.h"
#include "colloid_link.h"
#include "tests.h"

//__managed__ int fail;

__global__ void check_links_array_kernel(colloid_t *pc, int *fail) {
  int i, j, p, status;
  double rb[3];

  colloid_link_i(pc->links, 0, &i);
  colloid_link_j(pc->links, 0, &j);
  colloid_link_p(pc->links, 0, &p);
  colloid_link_status(pc->links, 0, &status);
  colloid_link_rb(pc->links, 0, rb);

  test_assert(i == pc->lnk->i);
  test_assert(j == pc->lnk->j);
  test_assert(p == pc->lnk->p);
  test_assert(status == pc->lnk->status);
  for (int index = 0; index < 3; index++)
    test_assert(rb[index] == pc->lnk->rb[index]);

  fail = 0;
} 

/*****************************************************************************
 *
 *  test_links_arrays_accessors
 *
 *****************************************************************************/

int test_device_memcpy_colloid_links_array(void) {
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  colloids_info_t * cinfo = NULL;
  colloid_options_t opts = colloid_options_default();
  colloid_t * pc;
  double r[3] = {0.5, 0.5, 0.5};

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);
  opts.have_colloids = 1;
  colloids_info_create(pe, cs, &opts, &cinfo);
  colloids_info_add_local(cinfo, 1, r, 1.0, &pc);
  pc->lnk = colloid_link_allocate();
  
  pc->lnk->i = 3;
  pc->lnk->j = 2;
  pc->lnk->p = 1;
  pc->lnk->status = 0;
  for (int i = 0; i < 3; i++)
    pc->lnk->rb[i] = i;

  copy_links_to_array(pc);

  colloids_memcpy(cinfo, tdpMemcpyHostToDevice);

  int fail = 1;
  //dim3 nblk(1,0,0), ntpb(1,0,0);
  //check_links_array_kernel<<<nblk, ntpb>>>(pc, &fail);

  if (fail == 0)
    pe_info(pe, "PASS     ./unit/test_links_arrays_accessors\n");

  return 0;
}