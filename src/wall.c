/*****************************************************************************
 *
 *  wall.c
 *
 *  Static solid objects (porous media).
 *
 *  Special case: boundary walls. The two issues might be sepatated.
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kernel.h"
#include "physics.h"
#include "util.h"
#include "wall.h"
#include "util_ellipsoid.h"
#include "util_vector.h"

typedef enum wall_init_enum {WALL_INIT_COUNT_ONLY,
			     WALL_INIT_ALLOCATE} wall_init_enum_t;

typedef enum wall_uw_enum {WALL_UZERO = 0,
			   WALL_UWTOP,
			   WALL_UWBOT,
			   WALL_UWMAX} wall_uw_enum_t;

int wall_init_boundaries(wall_t * wall, wall_init_enum_t init);
int wall_init_map(wall_t * wall);
int wall_init_uw(wall_t * wall);

__host__ int wall_init_boundaries_slip(wall_t * wall);
__host__ int wall_link_normal(wall_t * wall, int n, int wn[3]);
__host__ int wall_link_slip_direction(wall_t * wall, int n);
__host__ int wall_link_slip(wall_t * wall, int n);
__host__ int wall_memcpy_h2d(wall_t * wall);

__global__ void wall_setu_kernel(wall_t * wall, lb_t * lb);
__global__ void wall_bbl_kernel(wall_t * wall, lb_t * lb, map_t * map);
__global__ void wall_bbl_slip_kernel(wall_t * wall, lb_t * lb, map_t * map);

static __constant__ wall_param_t static_param;

/*****************************************************************************
 *
 *  wall_create
 *
 *****************************************************************************/

__host__ int wall_create(pe_t * pe, cs_t * cs, map_t * map, lb_t * lb,
			 wall_t ** p) {

  int ndevice;
  wall_t * wall = NULL;

  assert(pe);
  assert(cs);
  assert(lb);
  assert(p);

  wall = (wall_t *) calloc(1, sizeof(wall_t));
  assert(wall);
  if (wall == NULL) pe_fatal(pe, "calloc(wall_t) failed\n");

  wall->param = (wall_param_t *) calloc(1, sizeof(wall_param_t));
  if (wall->param == NULL) pe_fatal(pe, "calloc(wall_param_t) failed\n");

  wall->pe = pe;
  wall->cs = cs;
  wall->map = map;
  wall->lb = lb;

  cs_retain(cs);

  /* Target copy */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    wall->target = wall;
  }
  else {
    wall_param_t * tmp = NULL;

    tdpAssert( tdpMalloc((void **) &wall->target, sizeof(wall_t)) );
    tdpAssert( tdpMemset(wall->target, 0, sizeof(wall_t)) );
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(static_param));
    tdpAssert( tdpMemcpy(&wall->target->param, &tmp, sizeof(wall_param_t *),
			 tdpMemcpyHostToDevice) );
  }

  *p = wall;

  return 0;
}

/*****************************************************************************
 *
 *  wall_free
 *
 *****************************************************************************/

__host__ int wall_free(wall_t * wall) {

  assert(wall);

  if (wall->target != wall) {
    {
      int * tmp = NULL;
      tdpAssert( tdpMemcpy(&tmp, &wall->target->linki, sizeof(int *),
			   tdpMemcpyDeviceToHost) );
      tdpAssert( tdpFree(tmp) );
      tdpAssert( tdpMemcpy(&tmp, &wall->target->linkj, sizeof(int *),
			   tdpMemcpyDeviceToHost) );
      tdpAssert( tdpFree(tmp) );
      tdpAssert( tdpMemcpy(&tmp, &wall->target->linkp, sizeof(int *),
			   tdpMemcpyDeviceToHost) );
      tdpAssert( tdpFree(tmp) );
      tdpAssert( tdpMemcpy(&tmp, &wall->target->linku, sizeof(int *),
			   tdpMemcpyDeviceToHost) );
      tdpAssert( tdpFree(tmp) );
    }
    /* Release slip stuff */
    if (wall->param->slip.active) {
      int * tmp = NULL;
      tdpAssert(tdpMemcpy(&tmp, &wall->target->linkk, sizeof(int *),
			  tdpMemcpyDeviceToHost));
      tdpAssert( tdpFree(tmp) );
    }
    if (wall->param->slip.active) {
      int8_t * tmp = NULL;
      tdpAssert(tdpMemcpy(&tmp, &wall->target->linkq, sizeof(int8_t *),
			  tdpMemcpyDeviceToHost));
      tdpAssert( tdpFree(tmp) );
      tdpAssert(tdpMemcpy(&tmp, &wall->target->links, sizeof(int8_t *),
			  tdpMemcpyDeviceToHost));
      tdpAssert( tdpFree(tmp) );
    }
    tdpAssert( tdpFree(wall->target) );
  }

  cs_free(wall->cs);
  free(wall->param);

  /* slip quantities */

  free(wall->linkk);
  free(wall->linkq);
  free(wall->links);

  free(wall->linki);
  free(wall->linkj);
  free(wall->linkp);
  free(wall->linku);

  free(wall);

  return 0;
}

/*****************************************************************************
 *
 *  wall_commit
 *
 *****************************************************************************/

__host__ int wall_commit(wall_t * wall, wall_param_t * param) {

  assert(wall);
  assert(param);

  *wall->param = *param;

  wall_init_map(wall);
  wall_init_boundaries(wall, WALL_INIT_COUNT_ONLY);
  wall_init_boundaries(wall, WALL_INIT_ALLOCATE);
  wall_init_boundaries_slip(wall);
  wall_init_uw(wall);

  /* As we have initialised the map on the host, ... */
  map_memcpy(wall->map, tdpMemcpyHostToDevice);

  wall_memcpy(wall, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  wall_info
 *
 *  Note a global communication.
 *
 *****************************************************************************/

__host__ int wall_info(wall_t * wall) {

  int nlink;
  pe_t * pe = NULL;
  MPI_Comm comm;

  assert(wall);

  pe = wall->pe;

  pe_mpi_comm(pe, &comm);
  MPI_Reduce(&wall->nlink, &nlink, 1, MPI_INT, MPI_SUM, 0, comm);

  if (wall->param->iswall) {
    pe_info(pe, "\n");
    pe_info(pe, "Boundary walls\n");
    pe_info(pe, "--------------\n");
    pe_info(pe, "Boundary walls:                  %1s %1s %1s\n",
	    (wall->param->isboundary[X] == 1) ? "X" : "-",
	    (wall->param->isboundary[Y] == 1) ? "Y" : "-",
	    (wall->param->isboundary[Z] == 1) ? "Z" : "-");
    pe_info(pe, "Boundary speed u_x (bottom):    %14.7e\n",
	    wall->param->ubot[X]);
    pe_info(pe, "Boundary speed u_x (top):       %14.7e\n",
	    wall->param->utop[X]);
    pe_info(pe, "Boundary normal lubrication rc: %14.7e\n",
	    wall->param->lubr_rc[X]);
    /* Print only if non-zero for backwards compatibility in tests ...*/
    if (wall->param->lubr_dh[X] > 0.0) {
    pe_info(pe, "Boundary normal lubrication dh: %14.7e\n",
	    wall->param->lubr_dh[X]);
    }

    pe_info(pe, "Wall boundary links allocated:   %d\n",  nlink);
    pe_info(pe, "Memory (total, bytes):           %zu\n", 4*nlink*sizeof(int));
    pe_info(pe, "Boundary shear initialise:       %d\n",
	    wall->param->initshear);
  }

  if (wall->param->slip.active) {
    pe_info(pe, "Wall slip active:                %s\n", "yes");
    pe_info(pe, "Wall slip fraction (bottom):    %14.7e %14.7e %14.7e\n",
	    wall->param->slip.s[WALL_SLIP_XBOT],
	    wall->param->slip.s[WALL_SLIP_YBOT],
	    wall->param->slip.s[WALL_SLIP_ZBOT]);
    pe_info(pe, "Wall slip fraction (top):       %14.7e %14.7e %14.7e\n",
	    wall->param->slip.s[WALL_SLIP_XTOP],
	    wall->param->slip.s[WALL_SLIP_YTOP],
	    wall->param->slip.s[WALL_SLIP_ZTOP]);
    pe_info(pe, "Memory (total, bytes):           %zu\n",
	    nlink*(1*sizeof(int) + 2*sizeof(int8_t)));
  }

  if (wall->param->isporousmedia) {
    pe_info(pe, "\n");
    pe_info(pe, "Porous Media\n");
    pe_info(pe, "------------\n");
    pe_info(pe, "Wall boundary links allocated:   %d\n",  nlink);
    pe_info(pe, "Memory (total, bytes):           %zu\n", 4*nlink*sizeof(int));
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_target
 *
 *****************************************************************************/

__host__ int wall_target(wall_t * wall, wall_t ** target) {

  assert(wall);
  assert(target);

  *target = wall->target;

  return 0;
}

/*****************************************************************************
 *
 *  wall_slip
 *
 *  A convenience to load the slip lookup table.
 *
 *****************************************************************************/

__host__ wall_slip_t wall_slip(double sbot[3], double stop[3]) {

  wall_slip_t ws = {0};

  ws.active = (sbot[X] != 0.0 || sbot[Y] != 0.0 || sbot[Z] != 0.0 ||
               stop[X] != 0.0 || stop[Y] != 0.0 || stop[Z] != 0.0);

  ws.s[WALL_NO_SLIP]   = 0.0;
  ws.s[WALL_SLIP_XBOT] = sbot[X];
  ws.s[WALL_SLIP_XTOP] = stop[X];
  ws.s[WALL_SLIP_YBOT] = sbot[Y];
  ws.s[WALL_SLIP_YTOP] = stop[Y];
  ws.s[WALL_SLIP_ZBOT] = sbot[Z];
  ws.s[WALL_SLIP_ZTOP] = stop[Z];

  /* Unique edge values (average of two sides) */

  ws.s[WALL_SLIP_EDGE_XB_YB] = 0.5*(sbot[X] + sbot[Y]);
  ws.s[WALL_SLIP_EDGE_XB_YT] = 0.5*(sbot[X] + stop[Y]);
  ws.s[WALL_SLIP_EDGE_XB_ZB] = 0.5*(sbot[X] + sbot[Z]);
  ws.s[WALL_SLIP_EDGE_XB_ZT] = 0.5*(sbot[X] + stop[Z]);
  ws.s[WALL_SLIP_EDGE_XT_YB] = 0.5*(stop[X] + sbot[Y]);
  ws.s[WALL_SLIP_EDGE_XT_YT] = 0.5*(stop[X] + stop[Y]);
  ws.s[WALL_SLIP_EDGE_XT_ZB] = 0.5*(stop[X] + sbot[Z]);
  ws.s[WALL_SLIP_EDGE_XT_ZT] = 0.5*(stop[X] + stop[Z]);
  ws.s[WALL_SLIP_EDGE_YB_ZB] = 0.5*(sbot[Y] + sbot[Z]);
  ws.s[WALL_SLIP_EDGE_YB_ZT] = 0.5*(sbot[Y] + stop[Z]);
  ws.s[WALL_SLIP_EDGE_YT_ZB] = 0.5*(stop[Y] + sbot[Z]);
  ws.s[WALL_SLIP_EDGE_YT_ZT] = 0.5*(stop[Y] + stop[Z]);

  return ws;
}

/*****************************************************************************
 *
 *  wall_slip_valid
 *
 *****************************************************************************/

__host__ int wall_slip_valid(const wall_slip_t * ws) {

  int valid = 1;

  assert(ws);

  if (ws->s[WALL_NO_SLIP] != 0.0) valid = 0;

  for (int n = 1; n < WALL_SLIP_MAX; n++) {
    if (0.0 > ws->s[n] || ws->s[n] > 1.0) valid = 0;
  }

  return valid;
}

/*****************************************************************************
 *
 *  wall_param_set
 *
 *****************************************************************************/

__host__ int wall_param_set(wall_t * wall, wall_param_t * values) {

  assert(wall);
  assert(values);

  *wall->param = *values;

  return 0;
}

/*****************************************************************************
 *
 *  wall_param
 *
 *****************************************************************************/

__host__ int wall_param(wall_t * wall, wall_param_t * values) {

  assert(wall);
  assert(values);

  *values = *wall->param;

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_boundaries
 *
 *  To be called twice:
 *     1. with WALL_INIT_COUNT_ONLY
 *     2. with WALL_INIT_ALLOCATE
 *
 *****************************************************************************/

__host__ int wall_init_boundaries(wall_t * wall, wall_init_enum_t init) {

  int ic, jc, kc;
  int ic1, jc1, kc1;
  int indexi, indexj;
  int p;
  int nlink;
  int nlocal[3];
  int status;
  int ndevice;

  assert(wall);
  assert(wall->lb);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (init == WALL_INIT_ALLOCATE) {
    nlink = imax(1, wall->nlink); /* Avoid zero-sized allocations */
    assert(nlink > 0);
    wall->linki = (int *) calloc(nlink, sizeof(int));
    wall->linkj = (int *) calloc(nlink, sizeof(int));
    wall->linkp = (int *) calloc(nlink, sizeof(int));
    wall->linku = (int *) calloc(nlink, sizeof(int));
    assert(wall->linki);
    assert(wall->linkj);
    assert(wall->linkp);
    assert(wall->linku);
    if (wall->linki == NULL) pe_fatal(wall->pe,"calloc(wall->linki) failed\n");
    if (wall->linkj == NULL) pe_fatal(wall->pe,"calloc(wall->linkj) failed\n");
    if (wall->linkp == NULL) pe_fatal(wall->pe,"calloc(wall->linkp) failed\n");
    if (wall->linku == NULL) pe_fatal(wall->pe,"calloc(wall->linku) failed\n");
    if (ndevice > 0) {
      int tmp;
      tdpAssert( tdpMalloc((void **) &tmp, wall->nlink*sizeof(int)) );
      tdpAssert( tdpMemcpy(&wall->target->linki, &tmp, sizeof(int *),
			   tdpMemcpyHostToDevice) );
      tdpAssert( tdpMalloc((void **) &tmp, wall->nlink*sizeof(int)) );
      tdpAssert( tdpMemcpy(&wall->target->linkj, &tmp, sizeof(int *),
			   tdpMemcpyHostToDevice) );
      tdpAssert( tdpMalloc((void **) &tmp, wall->nlink*sizeof(int)) );
      tdpAssert( tdpMemcpy(&wall->target->linkp, &tmp, sizeof(int *),
			   tdpMemcpyHostToDevice) );
      tdpAssert( tdpMalloc((void **) &tmp, wall->nlink*sizeof(int)) );
      tdpAssert( tdpMemcpy(&wall->target->linku, &tmp, sizeof(int *),
			   tdpMemcpyHostToDevice) );
    }
  }

  nlink = 0;
  cs_nlocal(wall->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	lb_t * lb = wall->lb;

	indexi = cs_index(wall->cs, ic, jc, kc);
	map_status(wall->map, indexi, &status);
	if (status != MAP_FLUID) continue;

	/* Look for non-solid -> solid links */

	for (p = 1; p < lb->model.nvel; p++) {

	  ic1 = ic + lb->model.cv[p][X];
	  jc1 = jc + lb->model.cv[p][Y];
	  kc1 = kc + lb->model.cv[p][Z];
	  indexj = cs_index(wall->cs, ic1, jc1, kc1);
	  map_status(wall->map, indexj, &status);

	  if (status == MAP_BOUNDARY) {
	    if (init == WALL_INIT_ALLOCATE) {
	      wall->linki[nlink] = indexi;
	      wall->linkj[nlink] = indexj;
	      wall->linkp[nlink] = p;
	      wall->linku[nlink] = WALL_UZERO;
	    }
	    nlink += 1;
	  }
	}

	/* Next site */
      }
    }
  }

  wall->nlink = nlink;

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_boundaries_slip
 *
 *  The slip condition is slightly more complicated than the no-slip,
 *  as it involves an additional fluid site.
 *
 *  Further, some care should be taken that slip links occur in pairs;
 *  the pair should have a unique value of 's' (the fraction of slip)
 *  so that mass is conserved by the bbl.
 *
 *  This will examine the existing 'no-slip' links, and make necessary
 *  additions.
 *
 *****************************************************************************/

__host__ int wall_init_boundaries_slip(wall_t * wall) {

  int ndevice;

  assert(wall);
  assert(wall->cs);
  assert(wall->map);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (wall->param->slip.active) {

    int nlink;

    nlink = imax(1, wall->nlink); /* Avoid zero-sized allocations */
    assert(nlink > 0);
    wall->linkk = (int *) calloc(nlink, sizeof(int));
    wall->linkq = (int8_t *) calloc(nlink, sizeof(int8_t));
    wall->links = (int8_t *) calloc(nlink, sizeof(int8_t));
    assert(wall->linkk);
    assert(wall->linkq);
    assert(wall->links);
    if (wall->linkk == NULL) pe_fatal(wall->pe,"calloc(wall->linkk) failed\n");
    if (wall->linkq == NULL) pe_fatal(wall->pe,"calloc(wall->linkq) failed\n");
    if (wall->links == NULL) pe_fatal(wall->pe,"calloc(wall->links) failed\n");

    /* Allocate device memory */
    if (ndevice > 0) {
      int tmp;
      tdpAssert( tdpMalloc((void **) &tmp, nlink*sizeof(int)) );
      tdpAssert (tdpMemcpy(&wall->target->linkk, &tmp, sizeof(int *),
			   tdpMemcpyHostToDevice) );
    }
    if (ndevice > 0) {
      int8_t tmp;
      tdpAssert( tdpMalloc((void **) &tmp, nlink*sizeof(int8_t)) );
      tdpAssert( tdpMemcpy(&wall->target->linkq, &tmp, sizeof(int8_t *),
			   tdpMemcpyHostToDevice) );
      tdpAssert( tdpMalloc((void **) &tmp, nlink*sizeof(int8_t)) );
      tdpAssert( tdpMemcpy(&wall->target->links, &tmp, sizeof(int8_t *),
			   tdpMemcpyHostToDevice) );
    }

    /* For each existing fluid-to-solid link i->j with cv[p] ... */
    /* Where is k (source of slipping distribution which will arrive at i),
       and what are q and s? */

    /* The location of k depends on the local wall normal at the nominal
     * bounce-back position i + cv[p]dt/2 (which might be a side, an edge
     * or a corner), so we must
     * re-examine the map to find the normal/tangent direction */

    for (int n = 0; n < nlink; n++) {

      lb_t * lb = wall->lb;

      /* Identify the wall normal wn, and project cv[p] into the
       * plane orthogonal to the normal to find the tangential
       * direction which takes us to the fluid site paired with
       * the current link for slip. */

      int wn[3] = {0};        /* Wall normal */
      int wt[3] = {0};        /* Wall tangent */
      int ijk[3] = {0};       /* fluid site linki (ic,jc,kc) */
      int p = wall->linkp[n];
      int cvdotwn, modwn, modwt;

      cs_index_to_ijk(wall->cs, wall->linki[n], ijk);
      wall_link_normal(wall, n, wn);

      /* tangent = cv[p] - (cv[p].n/n.n) n */
      /* But we are in integers, so watch any division */

      cvdotwn = lb->model.cv[p][X]*wn[X] + lb->model.cv[p][Y]*wn[Y]
	      + lb->model.cv[p][Z]*wn[Z];
      modwn = wn[X]*wn[X] + wn[Y]*wn[Y] + wn[Z]*wn[Z];

      assert(cvdotwn == -1 || cvdotwn == -2 || cvdotwn == -3);
      assert(modwn == -cvdotwn);

      wt[X] = (lb->model.cv[p][X] - cvdotwn*wn[X]/modwn);
      wt[Y] = (lb->model.cv[p][Y] - cvdotwn*wn[Y]/modwn);
      wt[Z] = (lb->model.cv[p][Z] - cvdotwn*wn[Z]/modwn);
      modwt = wt[X]*wt[X] + wt[Y]*wt[Y] + wt[Z]*wt[Z];

      if (modwt == 0) {
	/* If there's no tangential component, p is not relevant for
	 * slip. Set WALL_NO_SLIP (k, q must be valid, but they
	 * don't enter the final result). */
	wall->linkk[n] = wall->linki[n];
	wall->linkq[n] = wall->linkp[n];
	wall->links[n] = WALL_NO_SLIP;
      }
      else {
	ijk[X] = ijk[X] + wt[X];
	ijk[Y] = ijk[Y] + wt[Y];
	ijk[Z] = ijk[Z] + wt[Z];
	wall->linkk[n] = cs_index(wall->cs, ijk[X], ijk[Y], ijk[Z]);
	wall->linkq[n] = wall_link_slip_direction(wall, n);
	wall->links[n] = wall_link_slip(wall, n);
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  wall_link_normal
 *
 *  For link n, what is the wall normal at the crossing position?
 *  This is for the special case iswall only.
 *
 *****************************************************************************/

__host__ int wall_link_normal(wall_t * wall, int n, int wn[3]) {

  assert(wall);
  assert(wall->param->iswall);
  assert(0 <= n && n < wall->nlink);

  wn[X] = 0;
  wn[Y] = 0;
  wn[Z] = 0;

  if (wall->param->iswall) {

    lb_t * lb = wall->lb;

    int i0[3] = {0};
    int p = wall->linkp[n];
    int ic, jc, kc, index;
    int status;

    cs_index_to_ijk(wall->cs, wall->linki[n], i0);

    ic = i0[X] + lb->model.cv[p][X]; jc = i0[Y], kc = i0[Z];
    index = cs_index(wall->cs, ic, jc, kc);
    map_status(wall->map, index, &status);
    if (status != MAP_FLUID) wn[X] = -lb->model.cv[p][X];

    ic = i0[X]; jc = i0[Y] + lb->model.cv[p][Y]; kc = i0[Z];
    index = cs_index(wall->cs, ic, jc, kc);
    map_status(wall->map, index, &status);
    if (status != MAP_FLUID) wn[Y] = -lb->model.cv[p][Y];

    ic = i0[X]; jc = i0[Y]; kc = i0[Z] + lb->model.cv[p][Z];
    index = cs_index(wall->cs, ic, jc, kc);
    map_status(wall->map, index, &status);
    if (status != MAP_FLUID) wn[Z] = -lb->model.cv[p][Z];
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_link_slip_direction
 *
 *  For given link n, work out the index q of the slip velocity
 *  for the relevant adjacent fluid site (which will be incoming
 *  at link fluid site i).
 *
 *  q = -1 is returned if no link is identified - an error.
 *  q  = 0 should also be regarded as erroneous.
 *
 *****************************************************************************/

__host__ int wall_link_slip_direction(wall_t * wall, int n) {

  int q = -1;

  assert(wall);
  assert(wall->param->iswall);
  assert(0 <= n && n < wall->nlink);

  if (wall->param->iswall) {

    lb_t * lb = wall->lb;

    int cq[3] = {0};
    int wn[3] = {0};
    int pn = wall->linkp[n];

    /* Components are reversed in the tangential direction, but
     * the same in the (-ve outward) normal direction. */

    wall_link_normal(wall, n, wn);

    cq[X] = -2*wn[X] - lb->model.cv[pn][X];
    cq[Y] = -2*wn[Y] - lb->model.cv[pn][Y];
    cq[Z] = -2*wn[Z] - lb->model.cv[pn][Z];

    /* Find the appropriate index */

    for (int p = 0; p < lb->model.nvel; p++) {
      if (cq[X] == lb->model.cv[p][X] &&
	  cq[Y] == lb->model.cv[p][Y] &&
	  cq[Z] == lb->model.cv[p][Z]) q = p;
    }
    assert(0 < q && q < lb->model.nvel);
  }

  return q;
}

/*****************************************************************************
 *
 *  wall_link_slip
 *
 *  For a given link n, find the relevant indirection into the slip value
 *  table to get the fraction of slip s.
 *
 *  Not this only considers the wall normal (not the link direction per se).
 *
 *****************************************************************************/

__host__ int wall_link_slip(wall_t * wall, int n) {

  int s = WALL_NO_SLIP;
  int wn[3] = {0};
  int modwn = 0;

  assert(wall);
  assert(0 <= n && n < wall->nlink);

  wall_link_normal(wall, n, wn);

  modwn = wn[X]*wn[X] + wn[Y]*wn[Y] + wn[Z]*wn[Z];

  switch (modwn) {
  case 1:
    /* A face (six cases) : set the relevant value */
    if (wn[X] == +1) s = WALL_SLIP_XBOT;
    if (wn[X] == -1) s = WALL_SLIP_XTOP;
    if (wn[Y] == +1) s = WALL_SLIP_YBOT;
    if (wn[Y] == -1) s = WALL_SLIP_YTOP;
    if (wn[Z] == +1) s = WALL_SLIP_ZBOT;
    if (wn[Z] == -1) s = WALL_SLIP_ZTOP;
    break;
  case 2:
    /* An edge (12 cases); there must be a unique value of s. */
    if (wn[X] ==  1 && wn[Y] ==  1) s = WALL_SLIP_EDGE_XB_YB;
    if (wn[X] ==  1 && wn[Y] == -1) s = WALL_SLIP_EDGE_XB_YT;
    if (wn[X] ==  1 && wn[Z] ==  1) s = WALL_SLIP_EDGE_XB_ZB;
    if (wn[X] ==  1 && wn[Z] == -1) s = WALL_SLIP_EDGE_XB_ZT;
    if (wn[X] == -1 && wn[Y] ==  1) s = WALL_SLIP_EDGE_XT_YB;
    if (wn[X] == -1 && wn[Y] == -1) s = WALL_SLIP_EDGE_XT_YT;
    if (wn[X] == -1 && wn[Z] ==  1) s = WALL_SLIP_EDGE_XT_ZB;
    if (wn[X] == -1 && wn[Z] == -1) s = WALL_SLIP_EDGE_XT_ZT;
    if (wn[Y] ==  1 && wn[Z] ==  1) s = WALL_SLIP_EDGE_YB_ZB;
    if (wn[Y] ==  1 && wn[Z] == -1) s = WALL_SLIP_EDGE_YB_ZT;
    if (wn[Y] == -1 && wn[Z] ==  1) s = WALL_SLIP_EDGE_YT_ZB;
    if (wn[Y] == -1 && wn[Z] == -1) s = WALL_SLIP_EDGE_YT_ZT;
    break;
  case 3:
    /* A corner (8 cases): all must be no slip */
    s = WALL_NO_SLIP;
    break;
  default:
    assert(0);
  }

  assert(s < WALL_SLIP_MAX);

  return s;
}

/*****************************************************************************
 *
 *  wall_memcpy
 *
 *****************************************************************************/

__host__ int wall_memcpy(wall_t * wall, tdpMemcpyKind flag) {

  int ndevice;

  assert(wall);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    assert(wall->target == wall);
  }
  else {

    switch (flag) {
    case tdpMemcpyHostToDevice:
      wall_memcpy_h2d(wall);
      break;
    case tdpMemcpyDeviceToHost:
      assert(0); /* Not required */
      break;
    default:
      pe_fatal(wall->pe, "Should definitely not be here\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_memcpy_h2d
 *
 *  Host -> Device copy. Slightly tedious.
 *
 *****************************************************************************/

__host__ int wall_memcpy_h2d(wall_t * wall) {

  int * tmp = NULL;
  int nlink = 0;

  assert(wall);

  nlink = wall->nlink;

  tdpAssert(tdpMemcpy(&wall->target->nlink, &wall->nlink, sizeof(int),
		      tdpMemcpyHostToDevice));
  tdpAssert(tdpMemcpy(wall->target->fnet, wall->fnet, 3*sizeof(double),
		      tdpMemcpyHostToDevice));

  /* In turn, linki, linkj, linkp, linku */
  tdpAssert(tdpMemcpy(&tmp, &wall->target->linki, sizeof(int *),
		      tdpMemcpyDeviceToHost));
  tdpAssert(tdpMemcpy(tmp, wall->linki, nlink*sizeof(int),
		      tdpMemcpyHostToDevice));

  tdpAssert(tdpMemcpy(&tmp, &wall->target->linkj, sizeof(int *),
		      tdpMemcpyDeviceToHost));
  tdpAssert(tdpMemcpy(tmp, wall->linkj, nlink*sizeof(int),
		      tdpMemcpyHostToDevice));

  tdpAssert(tdpMemcpy(&tmp, &wall->target->linkp, sizeof(int *),
		      tdpMemcpyDeviceToHost));
  tdpAssert(tdpMemcpy(tmp, wall->linkp, nlink*sizeof(int),
		      tdpMemcpyHostToDevice));

  tdpAssert(tdpMemcpy(&tmp, &wall->target->linku, sizeof(int *),
		      tdpMemcpyDeviceToHost));
  tdpAssert(tdpMemcpy(tmp, wall->linku, nlink*sizeof(int),
		      tdpMemcpyHostToDevice));

  /* Slip stuff k, q, s ... */
  if (wall->param->slip.active) {
    int8_t * tmp8 = NULL;
    /* linkk, linkq, links ... */
    tdpAssert(tdpMemcpy(&tmp, &wall->target->linkk, sizeof(int *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(tmp, wall->linkk, nlink*sizeof(int),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&tmp8, &wall->target->linkq, sizeof(int8_t *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(tmp8, wall->linkq, nlink*sizeof(int8_t),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&tmp8, &wall->target->links, sizeof(int8_t *),
			tdpMemcpyDeviceToHost));
    tdpAssert(tdpMemcpy(tmp8, wall->links, nlink*sizeof(int8_t),
			tdpMemcpyHostToDevice));
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_uw
 *
 *  Only the simple case of one set of walls is handled at present.
 *
 *****************************************************************************/

__host__ int wall_init_uw(wall_t * wall) {

  int n;
  int iw;
  int nwall;

  assert(wall);

  nwall = wall->param->isboundary[X] + wall->param->isboundary[Y]
    + wall->param->isboundary[Z];

  if (nwall == 1) {
    lb_t * lb = wall->lb;
    /* All links are either top or bottom */
    iw = -1;
    if (wall->param->isboundary[X]) iw = X;
    if (wall->param->isboundary[Y]) iw = Y;
    if (wall->param->isboundary[Z]) iw = Z;
    assert(iw == X || iw == Y || iw == Z);

    for (n = 0; n < wall->nlink; n++) {
      if (lb->model.cv[wall->linkp[n]][iw] == -1) wall->linku[n] = WALL_UWBOT;
      if (lb->model.cv[wall->linkp[n]][iw] == +1) wall->linku[n] = WALL_UWTOP;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_set_wall_distribution
 *
 *  Driver routine.
 *
 *****************************************************************************/

__host__ int wall_set_wall_distributions(wall_t * wall) {

  dim3 nblk, ntpb;

  assert(wall);
  assert(wall->target);

  if (wall->nlink == 0) return 0;

  kernel_launch_param(wall->nlink, &nblk, &ntpb);

  tdpLaunchKernel(wall_setu_kernel, nblk, ntpb, 0, 0,
		  wall->target, wall->lb->target);

  tdpAssert( tdpPeekAtLastError() );
  tdpAssert( tdpDeviceSynchronize() );

  return 0;
}

/*****************************************************************************
 *
 *  wall_setu_kernel
 *
 *  Set distribution at solid sites to reflect the solid body velocity.
 *  This allows 'solid-solid' exchange of distributions between wall
 *  and colloids.
 *
 *****************************************************************************/

__global__ void wall_setu_kernel(wall_t * wall, lb_t * lb) {

  int n;
  int p;                   /* Outward going component of link velocity */
  double fp;               /* f = w_p (rho0 + (1/cs2) u_a c_pa) No sdotq */
  double ux = 0.0;         /* No initialisation */
  LB_RCS2_DOUBLE(rcs2);

  assert(wall);
  assert(lb);

  for_simt_parallel(n, wall->nlink, 1) {

    p = lb->param->nvel - wall->linkp[n];
    fp = lb->param->wv[p]*(lb->param->rho0 + rcs2*ux*lb->param->cv[p][X]);
    lb_f_set(lb, wall->linkj[n], p, LB_RHO, fp);

  }

  return;
}

/*****************************************************************************
 *
 *  wall_bbl
 *
 *  Driver routine.
 *
 *****************************************************************************/

__host__ int wall_bbl(wall_t * wall) {

  dim3 nblk, ntpb;
  void (* kernel) (wall_t * wall, lb_t * lb, map_t * map);

  assert(wall);
  assert(wall->target);

  if (wall->nlink == 0) return 0;

  kernel = wall_bbl_kernel;
  if (wall->param->slip.active) kernel = wall_bbl_slip_kernel;

  /* Update kernel constants */
  tdpMemcpyToSymbol(tdpSymbol(static_param), wall->param,
		    sizeof(wall_param_t), 0, tdpMemcpyHostToDevice);

  kernel_launch_param(wall->nlink, &nblk, &ntpb);

  tdpLaunchKernel(kernel, nblk, ntpb, 0, 0,
		  wall->target, wall->lb->target, wall->map->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  return 0;
}

/*****************************************************************************
 *
 *  wall_bbl_kernel
 *
 *  Bounce-back on links for the walls.
 *  A reduction is required to tally the net momentum transfer.
 *
 *****************************************************************************/

__global__ void wall_bbl_kernel(wall_t * wall, lb_t * lb, map_t * map) {

  int n;
  int ib;
  int tid;
  double fxb, fyb, fzb;
  double uw[WALL_UWMAX][3];

  __shared__ double fx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_MAX_THREADS_PER_BLOCK];

  LB_RCS2_DOUBLE(rcs2);

  assert(wall);
  assert(lb);
  assert(map);

  /* Load the current wall velocities into the uw table */

  for (ib = 0; ib < 3; ib++) {
    uw[WALL_UZERO][ib] = 0.0;
    uw[WALL_UWTOP][ib] = wall->param->utop[ib];
    uw[WALL_UWBOT][ib] = wall->param->ubot[ib];
  }

  tid = threadIdx.x;

  fx[tid] = 0.0;
  fy[tid] = 0.0;
  fz[tid] = 0.0;

  for_simt_parallel(n, wall->nlink, 1) {

    int i, j, ij, ji, ia;
    int status;
    double rho, cdotu;
    double fp, fp0, fp1;
    double force;

    i  = wall->linki[n];         /* fluid */
    j  = wall->linkj[n];         /* solid */
    ij = wall->linkp[n];         /* Link index direction fluid->solid */
    ji = lb->param->nvel - ij;   /* Opposite direction index */
    ia = wall->linku[n];         /* Wall velocity lookup */

    cdotu = lb->param->cv[ij][X]*uw[ia][X] +
            lb->param->cv[ij][Y]*uw[ia][Y] +
            lb->param->cv[ij][Z]*uw[ia][Z];

    map_status(map, i, &status);

    if (status == MAP_COLLOID) {

      /* This matches the momentum exchange in colloid BBL. */
      /* This only affects the accounting (via anomaly, as below) */

      lb_f(lb, i, ij, LB_RHO, &fp0);
      lb_f(lb, j, ji, LB_RHO, &fp1);
      fp = fp0 + fp1;

      fx[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
      fy[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
      fz[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];
    }
    else {

      /* This is the momentum. To prevent accumulation of round-off
       * in the running total (fnet_), we subtract the equilibrium
       * wv[ij]. This is ok for walls where there are exactly
       * equal and opposite links at each side of the system. */

      lb_f(lb, i, ij, LB_RHO, &fp);
      lb_0th_moment(lb, i, LB_RHO, &rho);

      force = 2.0*fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;

      fx[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
      fy[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
      fz[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];

      fp = fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;
      lb_f_set(lb, j, ji, LB_RHO, fp);

      if (lb->param->ndist > 1) {
	/* Order parameter */
	lb_f(lb, i, ij, LB_PHI, &fp);
	lb_0th_moment(lb, i, LB_PHI, &rho);

	fp = fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;
	lb_f_set(lb, j, ji, LB_PHI, fp);
      }
    }
    /* Next link */
  }

  /* Reduction for momentum transfer */

  fxb = tdpAtomicBlockAddDouble(fx);
  fyb = tdpAtomicBlockAddDouble(fy);
  fzb = tdpAtomicBlockAddDouble(fz);

  if (tid == 0) {
    tdpAtomicAddDouble(&wall->fnet[X], fxb);
    tdpAtomicAddDouble(&wall->fnet[Y], fyb);
    tdpAtomicAddDouble(&wall->fnet[Z], fzb);
  }

  return;
}

/*****************************************************************************
 *
 *  wall_bbl_slip_kernel
 *
 *  Version which:
 *    1. allows slip
 *    2. does not allow wall velocity
 *
 *****************************************************************************/

__global__ void wall_bbl_slip_kernel(wall_t * wall, lb_t * lb, map_t * map) {

  int n;
  int tid;
  double fxb, fyb, fzb;

  __shared__ double fx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_MAX_THREADS_PER_BLOCK];

  assert(wall);
  assert(lb);
  assert(map);

  tid = threadIdx.x;

  fx[tid] = 0.0;
  fy[tid] = 0.0;
  fz[tid] = 0.0;

  for_simt_parallel(n, wall->nlink, 1) {

    int i, j, ij, ji;
    int status;
    double fp, fp0, fp1;

    i  = wall->linki[n];
    j  = wall->linkj[n];
    ij = wall->linkp[n];        /* Link index direction solid->fluid */
    ji = lb->param->nvel - ij;  /* Opposite direction index */

    map_status(map, i, &status);

    if (status == MAP_COLLOID) {

      /* This matches the momentum exchange in colloid BBL. */
      /* This only affects the accounting (via anomaly, as below) */

      lb_f(lb, i, ij, LB_RHO, &fp0);
      lb_f(lb, j, ji, LB_RHO, &fp1);
      fp = fp0 + fp1;

      fx[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
      fy[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
      fz[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];
    }
    else {

      int k, q;           /* Note promotion to int for linkk, linkq */
      double fi, fk, s;
      double wx, wy, wz;

      k = wall->linkk[n];
      q = wall->linkq[n];
      s = wall->param->slip.s[wall->links[n]];

      lb_f(lb, i, ij, LB_RHO, &fi);
      lb_f(lb, k, q,  LB_RHO, &fk);
      fp = (1.0-s)*fi + s*fk;

      lb_f_set(lb, j, ji, LB_RHO, fp);

      /* Momentum change: no-slip has full contribution,
       * slip only contributes in wall normal direction
       * (wx, wy, wz) and we need |normal| to get sign. */

      fx[tid] += 2.0*(1.0-s)*(fi-lb->param->wv[ij])*lb->param->cv[ij][X];
      fy[tid] += 2.0*(1.0-s)*(fi-lb->param->wv[ij])*lb->param->cv[ij][Y];
      fz[tid] += 2.0*(1.0-s)*(fi-lb->param->wv[ij])*lb->param->cv[ij][Z];

      wx = -(lb->param->cv[ij][X] + lb->param->cv[q][X])/2;
      wy = -(lb->param->cv[ij][Y] + lb->param->cv[q][Y])/2;
      wz = -(lb->param->cv[ij][Z] + lb->param->cv[q][Z])/2;
      fx[tid] += 2.0*wx*wx*s*(fk-lb->param->wv[q])*lb->param->cv[q][X];
      fy[tid] += 2.0*wy*wy*s*(fk-lb->param->wv[q])*lb->param->cv[q][Y];
      fz[tid] += 2.0*wz*wz*s*(fk-lb->param->wv[q])*lb->param->cv[q][Z];
    }
    /* Next link */
  }

  /* Reduction for momentum transfer */

  fxb = tdpAtomicBlockAddDouble(fx);
  fyb = tdpAtomicBlockAddDouble(fy);
  fzb = tdpAtomicBlockAddDouble(fz);

  if (tid == 0) {
    tdpAtomicAddDouble(&wall->fnet[X], fxb);
    tdpAtomicAddDouble(&wall->fnet[Y], fyb);
    tdpAtomicAddDouble(&wall->fnet[Z], fzb);
  }

  return;
}

/*****************************************************************************
 *
 *  wall_init_map
 *
 *****************************************************************************/

__host__ int wall_init_map(wall_t * wall) {

  int ic, jc, kc, index;
  int ic_global, jc_global, kc_global;
  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  int nextra;

  assert(wall);

  cs_ntotal(wall->cs, ntotal);
  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);
  cs_nhalo(wall->cs, &nextra);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	/* If this is an appropriate periodic boundary, set to solid */

	ic_global = ic + noffset[X];
	jc_global = jc + noffset[Y];
	kc_global = kc + noffset[Z];

	if (wall->param->isboundary[Z]) {
	  if (kc_global == 0 || kc_global == ntotal[Z] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[Y]) {
	  if (jc_global == 0 || jc_global == ntotal[Y] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[X]) {
	  if (ic_global == 0 || ic_global == ntotal[X] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}
	/* next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum_add
 *
 *****************************************************************************/

__host__ int wall_momentum_add(wall_t * wall, const double f[3]) {

  assert(wall);

  wall->fnet[X] += f[X];
  wall->fnet[Y] += f[Y];
  wall->fnet[Z] += f[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum
 *
 *  If a global total is required, the caller is responsible for
 *  any MPI reduction. This is the local contribution.
 *
 *****************************************************************************/

__host__ int wall_momentum(wall_t * wall, double f[3]) {

  int ndevice;
  double ftmp[3];

  assert(wall);

  /* Some care at the moment. Accumulate the device total to the
   * host and zero the device fnet so that we don't double-count
   * it next time. */

  /* This is required as long as some contributions are made on
   * the host via wall_momentum_add() and others are on the
   * device. */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    tdpAssert( tdpMemcpy(ftmp, wall->target->fnet, 3*sizeof(double),
			 tdpMemcpyDeviceToHost) );
    wall->fnet[X] += ftmp[X];
    wall->fnet[Y] += ftmp[Y];
    wall->fnet[Z] += ftmp[Z];
    ftmp[X] = 0.0; ftmp[Y] = 0.0; ftmp[Z] = 0.0;
    tdpAssert( tdpMemcpy(wall->target->fnet, ftmp, 3*sizeof(double),
			 tdpMemcpyHostToDevice) );
  }

  /* Return the current net */

  f[X] = wall->fnet[X];
  f[Y] = wall->fnet[Y];
  f[Z] = wall->fnet[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_is_pm
 *
 *****************************************************************************/

__host__ __device__ int wall_is_pm(wall_t * wall, int * ispm) {

  assert(wall);

  *ispm = wall->param->isporousmedia;

  return 0;
}

/*****************************************************************************
 *
 *  wall_present
 *
 *****************************************************************************/

__host__ __device__ int wall_present(wall_t * wall) {

  wall_param_t * wp = NULL;

  assert(wall);

  wp = wall->param;
  return (wp->isboundary[X] || wp->isboundary[Y] || wp->isboundary[Z]);
}

/*****************************************************************************
 *
 *  wall_present_dim
 *
 *****************************************************************************/

__host__ __device__ int wall_present_dim(wall_t * wall, int iswall[3]) {

  assert(wall);

  iswall[X] = wall->param->isboundary[X];
  iswall[Y] = wall->param->isboundary[Y];
  iswall[Z] = wall->param->isboundary[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_shear_init
 *
 *  Initialise the distributions to be consistent with a linear shear
 *  profile for the given top and bottom wall velocities.
 *
 *  This is only relevant for walls at z = 0 and z = L_z.
 *  [Test coverage?]
 *
 *****************************************************************************/

__host__ int wall_shear_init(wall_t * wall) {

  int ic, jc, kc, index;
  int ia, ib, p;
  int nlocal[3];
  int noffset[3];
  double rho, u[3], gradu[3][3];
  double eta;
  double gammadot;
  double f;
  double cdotu;
  double sdotq;
  double uxbottom;
  double uxtop;
  double ltot[3];

  KRONECKER_DELTA_CHAR(d_);

  physics_t * phys = NULL;

  assert(wall);

  /* One wall constraint */
  uxtop = wall->param->utop[X];
  uxbottom = wall->param->ubot[X];

  cs_ltot(wall->cs, ltot);

  /* Shear rate */
  gammadot = (uxtop - uxbottom)/ltot[Z];

  pe_info(wall->pe, "Initialising linear shear profile for walls\n");
  pe_info(wall->pe, "Speed at top u_x    %14.7e\n", uxtop);
  pe_info(wall->pe, "Speed at bottom u_x %14.7e\n", uxbottom);
  pe_info(wall->pe, "Overall shear rate  %14.7e\n", gammadot);

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  physics_ref(&phys);
  physics_rho0(phys, &rho);
  physics_eta_shear(phys, &eta);

  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = 0.0;
    for (ib = 0; ib < 3; ib++) {
      gradu[ia][ib] = 0.0;
    }
  }

  /* Shear rate */
  gradu[X][Z] = gammadot;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	lb_t * lb = wall->lb;
	double cs2 = lb->model.cs2;
	double rcs2 = 1.0/cs2;

	/* Linearly interpolate between top and bottom to get velocity;
	 * the - 1.0 accounts for kc starting at 1. */
	u[X] = uxbottom + (noffset[Z] + kc - 0.5)*(uxtop - uxbottom)/ltot[Z];

        index = cs_index(wall->cs, ic, jc, kc);

        for (p = 0; p < lb->model.nvel; p++) {

	  cdotu = 0.0;
	  sdotq = 0.0;

          for (ia = 0; ia < 3; ia++) {
            cdotu += lb->model.cv[p][ia]*u[ia];
            for (ib = 0; ib < 3; ib++) {
              double q_ab = lb->model.cv[p][ia]*lb->model.cv[p][ib] - cs2*d_[ia][ib];
              sdotq += (rho*u[ia]*u[ib] - eta*gradu[ia][ib])*q_ab;
            }
          }
          f = lb->model.wv[p]*rho*(1.0 + rcs2*cdotu + 0.5*rcs2*rcs2*sdotq);
          lb_f_set(wall->lb, index, p, 0, f);
        }
        /* Next site */
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  wall_lubrication
 *
 *  This returns the normal lubrication correction for colloid of hydrodynamic
 *  radius ah at position r near a flat wall in dimension dim (if present).
 *  This is based on the analytical expression for a sphere.
 *
 *  The result should be added to the appropriate diagonal element of
 *  the colloid's drag matrix in the implicit update. There is, therefore,
 *  no velocity appearing here (wall assumed to have no velocity).
 *  This is therefore closely related to BBL in bbl.c.
 *
 *  This operates in parallel by computing the absolute distance between
 *  the side of the system (walls nominally at Lmin and (Lmax + Lmin)),
 *  and applying the cutoff.
 *
 *  An offset lubr_dh is available at each end [defaults to zero].
 *
 *  Normal force is added to the diagonal of drag matrix \zeta^FU_xx etc
 *  (No tangential force would be added to \zeta^FU_xx and \zeta^FU_yy)
 *
 *****************************************************************************/

__host__ int wall_lubr_sphere(wall_t * wall, double ah, const double r[3],
			      double drag[3]) {

  double eta;
  double lmin[3];
  double ltot[3];
  physics_t * phys = NULL;

  drag[X] = 0.0;
  drag[Y] = 0.0;
  drag[Z] = 0.0;

  if (wall == NULL) return 0;

  cs_lmin(wall->cs, lmin);
  cs_ltot(wall->cs, ltot);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);

  /* Lower, then upper wall X, Y, and Z */

  if (wall->param->isboundary[X]) {
    double dh = wall->param->lubr_dh[X];
    double hlub = wall->param->lubr_rc[X];
    double hb = r[X] - (lmin[X] + dh)  - ah;
    double ht = lmin[X] + (ltot[X] - dh) - r[X] - ah;
    drag[X] += wall_lubr_drag(eta, ah, hb, hlub);
    drag[X] += wall_lubr_drag(eta, ah, ht, hlub);
  }

  if (wall->param->isboundary[Y]) {
    double dh = wall->param->lubr_dh[Y];
    double hlub = wall->param->lubr_rc[Y];
    double hb = r[Y] - (lmin[Y] + dh)  - ah;
    double ht = lmin[Y] + (ltot[Y] - dh) - r[Y] - ah;
    drag[Y] += wall_lubr_drag(eta, ah, hb, hlub);
    drag[Y] += wall_lubr_drag(eta, ah, ht, hlub);
  }

  if (wall->param->isboundary[Z]) {
    double dh = wall->param->lubr_dh[Z];
    double hlub = wall->param->lubr_rc[Z];
    double hb = r[Z] - (lmin[Z] + dh) - ah;
    double ht = lmin[Z] + (ltot[Z] - dh) - r[Z] - ah;
    drag[Z] += wall_lubr_drag(eta, ah, hb, hlub);
    drag[Z] += wall_lubr_drag(eta, ah, ht, hlub);
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_lubr_drag
 *
 *  Return drag correction for sphere hydrodynamic radius ah.
 *  The separation h should be less than the cut off hc, or
 *  else zero is returned. Both h and hc should be +ve.
 *
 *  eta is the dynamic viscosity (lattice units).
 *
 *****************************************************************************/

__host__ double wall_lubr_drag(double eta, double ah, double h, double hc) {

  double zeta = 0.0;
  PI_DOUBLE(pi);

  if (h < hc) zeta = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hc);
  return zeta;
}
