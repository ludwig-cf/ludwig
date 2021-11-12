/*****************************************************************************
 *
 *  lb_bc_inflow_rhou.c
 *
 *  Implementation of an inflowboundary condition which assumes
 *  a channel geometry with walls in two dimensions.
 *
 *  The inflow in the flow direction is then at the left or lower
 *  end of the system and sets, e.g.,:
 *
 *    rho_inflow(x=0,y,z) = rho(x=1,y,z)     the adjacent point
 *    u_inflow(x=0,y,z)   = u0               a prescribed value
 *
 *  Incoming distributions are then set f_i = f^eq_i(rho, u) in
 *  the inflow boundary region.
 *
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
#include "lb_bc_inflow_rhou.h"

typedef enum {LINK_COUNT, LINK_ASSIGN} link_init_enum_t;

__host__ int lb_bc_inflow_rhou_init_internal(lb_bc_inflow_rhou_t * inflow);
__host__ int lb_bc_inflow_init_link(lb_bc_inflow_rhou_t * inflow,
				    link_init_enum_t flag, int id);

static int int_max(int a, int b) {return (a > b) ? a : b;}

static const lb_bc_open_vtable_t vt_ = {
  (lb_bc_open_free_ft)   lb_bc_inflow_rhou_free,
  (lb_bc_open_update_ft) lb_bc_inflow_rhou_update,
  (lb_bc_open_impose_ft) lb_bc_inflow_rhou_impose,
  (lb_bc_open_stats_ft)  lb_bc_inflow_rhou_stats
};

/*****************************************************************************
 *
 *  lb_bc_inflow_rhou_create
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_create(pe_t * pe, cs_t * cs,
				      const lb_openbc_options_t * options,
				      lb_bc_inflow_rhou_t ** inflow) {
  lb_bc_inflow_rhou_t * bc = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(inflow);

  bc = (lb_bc_inflow_rhou_t *) calloc(1, sizeof(lb_bc_inflow_rhou_t));
  assert(bc);
  if (bc == NULL) pe_fatal(pe, "Failed to allocate lb_bc_inflow_rhou_t");

  /* Pointers; superclass block */
  bc->pe = pe;
  bc->cs = cs;

  bc->super.func = &vt_;
  bc->super.id   = LB_BC_INFLOW_RHOU;

  if (!lb_openbc_options_valid(options)) {
    /* Internal error if we reach this point. */
    pe_fatal(pe, "lb_bc_inflow_rhou_create: lb_openbc_options_t invalid\n");
  }
  bc->options = *options;

  lb_bc_inflow_rhou_init_internal(bc);

  *inflow = bc;

  return 0;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_rhou_init_internal
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_init_internal(lb_bc_inflow_rhou_t * inflow) {

  assert(inflow);

  if (inflow->options.flow[X]) lb_bc_inflow_init_link(inflow, LINK_COUNT, X);
  if (inflow->options.flow[Y]) lb_bc_inflow_init_link(inflow, LINK_COUNT, Y);
  if (inflow->options.flow[Z]) lb_bc_inflow_init_link(inflow, LINK_COUNT, Z);

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

  if (inflow->options.flow[X]) lb_bc_inflow_init_link(inflow, LINK_ASSIGN, X);
  if (inflow->options.flow[Y]) lb_bc_inflow_init_link(inflow, LINK_ASSIGN, Y);
  if (inflow->options.flow[Z]) lb_bc_inflow_init_link(inflow, LINK_ASSIGN, Z);

  return 0;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_init_link
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

__host__ int lb_bc_inflow_init_link(lb_bc_inflow_rhou_t * inflow,
				    link_init_enum_t init, int id) {

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

	    if (init == LINK_ASSIGN) {
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
 *  lb_bc_inflow_rhou_free
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_free(lb_bc_inflow_rhou_t * inflow) {

  assert(inflow);

  free(inflow->linkj);
  free(inflow->linki);
  free(inflow->linkp);
  free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_rhou_update
 *
 *  Set any relevant hydrodynamic conditions at the inflow:
 *    (1) rho is taken from directly adjacent site in domain;
 *    (2) u is set to the prescribed constant value.
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_update(lb_bc_inflow_rhou_t * inflow,
				      hydro_t * hydro) {

  cs_t * cs = NULL;
  int id = -1;
  int noffset[3] = {};

  assert(inflow);
  assert(hydro);

  cs = inflow->cs;
  cs_nlocal_offset(cs, noffset);

  if (inflow->options.flow[X]) id = X;
  if (inflow->options.flow[Y]) id = Y;
  if (inflow->options.flow[Z]) id = Z;


  if (noffset[id] == 0) {
    int nlocal[3] = {};
    int idx = inflow->options.flow[X];
    int jdy = inflow->options.flow[Y];
    int kdz = inflow->options.flow[Z];

    cs_nlocal(cs, nlocal);

    nlocal[id] = 1; /* Only leftmost edge in the relevant direction */

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {

	  int index0 = cs_index(cs, ic - idx, jc - jdy, kc - kdz);
	  int index1 = cs_index(cs, ic, jc, kc);
	  double rho = 1.0;

	  hydro_rho(hydro, index1, &rho);
	  hydro_rho_set(hydro, index0, rho);
	  hydro_u_set(hydro, index0, inflow->options.u0);
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_rhou_impose
 *
 *  Intent: After lattice halo swap; before propagation.
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_impose(lb_bc_inflow_rhou_t * inflow,
				      hydro_t * hydro,
				      lb_t * lb) {
  assert(inflow);
  assert(hydro);
  assert(lb);

  /* For each incoming link set f_p = f^eq_p (rho, u) */

  for (int n = 0; n < inflow->nlink; n++) {

    int index = inflow->linkj[n];
    int8_t q  = lb->model.nvel - inflow->linkp[n];

    double rho  = 0.0;
    double u[3] = {};

    hydro_rho(hydro, index, &rho);
    hydro_u(hydro, index, u);

    {
      double cs2   = lb->model.cs2;
      double rcs2  = (1.0/cs2);
      double udotc = 0.0;
      double sdotq = 0.0;
      double fp    = 0.0;

      for (int ia = 0; ia < 3; ia++) {
	udotc += u[ia]*lb->model.cv[q][ia];
	for (int ib = 0; ib < 3; ib++) {
	  double d_ab = (ia == ib);
	  double s_ab = lb->model.cv[q][ia]*lb->model.cv[q][ib] - cs2*d_ab;

	  sdotq += s_ab*u[ia]*u[ib];
	}
      }

      /* Here's the equilibrium */
      fp = rho*lb->model.wv[q]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
      lb_f_set(lb, index, q, LB_RHO, fp);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_bc_inflow_rhou_stats
 *
 *  Placeholder. No operation at the moment.
 *
 *****************************************************************************/

__host__ int lb_bc_inflow_rhou_stats(lb_bc_inflow_rhou_t * inflow) {

  return 0;
}
