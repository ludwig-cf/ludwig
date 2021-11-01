/*****************************************************************************
 *
 *  lb_inflow_rhou.c
 *
 *  Concrete instance of lb_open_bc.h for inflow.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "lb_model.h"
#include "lb_inflow_rhou.h"

typedef enum {LB_LINK_COUNT, LB_LINK_ASSIGN} lb_link_init_enum_t;

__host__ int lb_inflow_rhou_init_internal(lb_inflow_rhou_t * inflow);
__host__ int lb_inflow_init_link(lb_inflow_rhou_t * inflow,
				 lb_link_init_enum_t flag, int id);

static int int_max(int a, int b) {return (a > b) ?a :b;}


static const lb_open_bc_vtable_t vt_ = {
  (lb_open_bc_free_ft)   lb_inflow_rhou_free,
  (lb_open_bc_update_ft) lb_inflow_rhou_update,
  (lb_open_bc_stats_ft)  lb_inflow_rhou_stats
};

/*****************************************************************************
 *
 *  lb_inflow_rhou_create
 *
 *****************************************************************************/

__host__ int lb_inflow_rhou_create(pe_t * pe, cs_t * cs,
				    const lb_openbc_options_t * options,
				    lb_inflow_rhou_t ** inflow) {
  lb_inflow_rhou_t * bc = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(inflow);

  bc = (lb_inflow_rhou_t *) calloc(1, sizeof(lb_inflow_rhou_t));
  assert(bc);
  if (bc == NULL) pe_fatal(pe, "Failed to allocate lb_inflow_rhou_t");

  /* Pointers; superclass block */
  bc->pe = pe;
  bc->cs = cs;

  bc->super.func = &vt_;
  bc->super.id   = LB_OPEN_BC_INFLOW_RHOU;

  if (!lb_openbc_options_valid(options)) {
    /* Internal error if we reach this point. */
    pe_fatal(pe, "lb_inflow_rhou_create: lb_openbc_options_t invalid\n");
  }
  bc->options = *options;

  lb_inflow_rhou_init_internal(bc);

  *inflow = bc;

  return 0;
}

/*****************************************************************************
 *
 *  lb_inflow_rhou_init_internal
 *
 *****************************************************************************/

__host__ int lb_inflow_rhou_init_internal(lb_inflow_rhou_t * inflow) {

  assert(inflow);

  if (inflow->options.flow[X]) lb_inflow_init_link(inflow, LB_LINK_COUNT, X);
  if (inflow->options.flow[Y]) lb_inflow_init_link(inflow, LB_LINK_COUNT, Y);
  if (inflow->options.flow[Z]) lb_inflow_init_link(inflow, LB_LINK_COUNT, Z);

  {
    pe_t * pe = inflow->pe;
    int nlink = int_max(1, inflow->nlink); /* No zero sized allocations. */

    inflow->linkp = (int8_t *) calloc(nlink, sizeof(int8_t));
    inflow->linki = (int *)    calloc(nlink, sizeof(int));
    inflow->linkj = (int *)    calloc(nlink, sizeof(int));

    assert(inflow->linkp);
    assert(inflow->linki);
    assert(inflow->linkj);

    if (inflow->linkp == NULL) pe_fatal(pe, "calloc(inflow->linkp) NULL\n");
    if (inflow->linki == NULL) pe_fatal(pe, "calloc(inflow->linki) NULL\n");
    if (inflow->linkj == NULL) pe_fatal(pe, "calloc(inflow->linkj) NULL\n");
  }

  if (inflow->options.flow[X]) lb_inflow_init_link(inflow, LB_LINK_ASSIGN, X);
  if (inflow->options.flow[Y]) lb_inflow_init_link(inflow, LB_LINK_ASSIGN, Y);
  if (inflow->options.flow[Z]) lb_inflow_init_link(inflow, LB_LINK_ASSIGN, Z);

  return 0;
}

/*****************************************************************************
 *
 *  lb_inflow_init_link
 *
 *  Identify links representing incoming distributions at the inflow
 *  which are fluid to fluid in the coordinate direction "id".
 *
 *  This assumes that the other two directions are walls, and so no
 *  duplicates of incoming solid-fluid (bbl) links are wanted.
 *
 *  CHECK CHECK CHECK links are fluid (domain) to fluid (halo). The
 *  incoming distributions are then actually the complements -cv[p].
 *
 *****************************************************************************/

__host__ int lb_inflow_init_link(lb_inflow_rhou_t * inflow,
				 lb_link_init_enum_t init, int id) {

  cs_t * cs = NULL;
  int noffset[3] = {};
  int nlink = 0;

  assert(inflow);
  assert(id == X || id == Y || id == Z);

  cs = inflow->cs;
  cs_nlocal_offset(cs, noffset);

  if (noffset[id] == 0) {
    int ntotal[3] = {};
    int nlocal[3] = {};

    lb_model_t model = {};
    lb_model_create(inflow->options.nvel, &model);

    cs_ntotal(cs, ntotal);
    cs_nlocal(cs, nlocal);

    nlocal[id] = 1; /* Only leftmost edge in the relevant direction */

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {

	  for (int p = 1; p < model.nvel; p++) {

	    if (model.cv[p][id] != -1) continue;

	    int ic1 = ic + model.cv[p][X];
	    int jc1 = jc + model.cv[p][Y];
	    int kc1 = kc + model.cv[p][Z];

	    /* Some shuffling to get "other 2 directions" */
	    int id1 = (id + 1) % 3;
	    int id2 = (id + 2) % 3;

	    int ijk[3] = {ic1, jc1, kc1};

	    if (noffset[id1] + ijk[id1] < 1          ) continue;
	    if (noffset[id1] + ijk[id1] > ntotal[id1]) continue;
	    if (noffset[id2] + ijk[id2] < 1          ) continue;
	    if (noffset[id2] + ijk[id2] > ntotal[id2]) continue;

	    if (init == LB_LINK_ASSIGN) {
	      inflow->linkp[nlink] = p;
	      inflow->linki[nlink] = cs_index(cs, ic, jc, kc);
	      inflow->linkj[nlink] = cs_index(cs, ic1, jc1, kc1);
	    }
	    nlink += 1;
	  }
	}
      }
    }

    lb_model_free(&model);
  }

  inflow->nlink = nlink;

  return 0;
}

/*****************************************************************************
 *
 *  lb_inflow_rhou_free
 *
 *****************************************************************************/

__host__ int lb_inflow_rhou_free(lb_inflow_rhou_t * inflow) {

  assert(inflow);

  free(inflow->linkj);
  free(inflow->linki);
  free(inflow->linkp);
  free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  lb_inflow_rhou_update
 *
 *  Drive the update of (i.e., apply) the boundary conditions.
 *
 *  1. We must update the rho u in the halo region post collision
 *     - Allows any relevant f_i(rho, u) to be computed.
 *     - Allows correct u to enter any flux calculation at edge of grid.
 *  2. Update post collision distributions in halo region appropriately
 *     to enter propagation stage.
 *
 *  After lattice halo swap; before propagation.
 *
 *****************************************************************************/

__host__ int lb_inflow_rhou_update(lb_inflow_rhou_t * inflow, hydro_t * hydro,
				   lb_t * lb) {

  assert(inflow);
  assert(hydro);
  assert(lb);

  /* Potentially both at the same time, as independent */

  /* Inflow */
  /* Simple for each link: */
  /*     compute relevant rho = rho(i); u0 fixed */
  /*     set appropriate f^eq_p(rho, u0) at j */

  /* Less simple for each link: */
  /*      compute relevant rho = rho(i) post collision */
  /*      compute reelvant u0 = u0(r_j) set by boundary position */
  /*      set appropriate f^eq_p(rho, u0) */

  /* Outflow */

  /* Simple for each link: */
  /*      compute relevant rho = rho0 */
  /*      compute reelvant  u0 = u0(r_i) */
  /*      set appropriate f^eq_p(rho, u0) */

  return 0;
}

/*****************************************************************************
 *
 *  lb_inflow_rhou_stats
 *
 *  No operation at the moment.
 *
 *****************************************************************************/

__host__ int lb_inflow_rhou_stats(lb_inflow_rhou_t * inflow) {

  return 0;
}
