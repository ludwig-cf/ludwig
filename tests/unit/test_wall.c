/*****************************************************************************
 *
 *  test_wall.c
 *
 *  Tests for flat wall special case.
 *
 *  (Porous media should be considered as a separate case).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2022 The University of Edinburgh
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
#include "wall.h"
#include "util.h"
#include "tests.h"

__host__ int wall_link_normal(wall_t * wall, int n, int wn[3]);
__host__ int wall_link_slip_direction(wall_t * wall, int n);
__host__ int wall_link_slip(wall_t * wall, int n);

__host__ int test_wall_slip(void);
__host__ int test_wall_link_normal(pe_t * pe, cs_t * cs);
__host__ int test_wall_link_slip_direction(pe_t * pe, cs_t * cs);
__host__ int test_wall_link_slip(pe_t * pe, cs_t * cs);
__host__ int test_wall_commit1(pe_t * pe, cs_t * cs);
__host__ int test_wall_commit2(pe_t * pe, cs_t * cs);

__host__ int test_wall_lubr_drag(void);

/*****************************************************************************
 *
 *  test_wall_suite
 *
 *****************************************************************************/

__host__ int test_wall_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  {
    int ntotal[3] = {8, 8, 8};
    cs_ntotal_set(cs, ntotal);
  }
  cs_init(cs);

  test_wall_slip();

  test_wall_link_normal(pe, cs);
  test_wall_link_slip_direction(pe, cs);
  test_wall_link_slip(pe, cs);

  test_wall_commit1(pe, cs);
  test_wall_commit2(pe, cs);

  test_wall_lubr_drag();

  pe_info(pe, "PASS     ./unit/test_wall\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_slip
 *
 *****************************************************************************/

__host__ int test_wall_slip(void) {

  wall_slip_t ws = {0};
  int valid = 0;

  valid = wall_slip_valid(&ws);
  assert(valid);

  {
    /* These are invalid values for fraction of slip... */
    /* ... but just for testing different numbers: */
    double sbot[3] = {1.0,  2.0,  3.0};
    double stop[3] = {4.0,  5.0,  6.0};

    ws = wall_slip(sbot, stop);
    assert(fabs(ws.s[WALL_NO_SLIP]   - 0.0)     < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_XBOT] - sbot[X]) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_XTOP] - stop[X]) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_YBOT] - sbot[Y]) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_YTOP] - stop[Y]) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_ZBOT] - sbot[Z]) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_ZTOP] - stop[Z]) < DBL_EPSILON);

    assert(ws.active);
    assert(!wall_slip_valid(&ws));
  }

  {
    /* ... and these are valid */
    double sbot[3] = {0.0, 0.5, 1.0};
    double stop[3] = {0.0, 0.5, 1.0};

    ws = wall_slip(sbot, stop);
    assert(ws.active);
    assert(wall_slip_valid(&ws));
  }

  /* Edges: test some reasonable cases */
  {
    double sbot[3] = {0.1, 0.2, 0.0};
    double stop[3] = {0.5, 0.7, 0.0};

    ws = wall_slip(sbot, stop);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XB_YB] - 0.15) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XB_YT] - 0.40) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XT_YB] - 0.35) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XT_YT] - 0.60) < DBL_EPSILON);
  }

  {
    double sbot[3] = {0.2, 0.0, 0.3};
    double stop[3] = {0.4, 0.0, 0.7};

    ws = wall_slip(sbot, stop);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XB_ZB] - 0.25) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XB_ZT] - 0.45) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XT_ZB] - 0.35) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_XT_ZT] - 0.55) < DBL_EPSILON);
  }

  {
    double sbot[3] = {0.0, 0.5, 0.3};
    double stop[3] = {0.0, 0.1, 0.4};

    ws = wall_slip(sbot, stop);
    assert(fabs(ws.s[WALL_SLIP_EDGE_YB_ZB] - 0.40) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_YB_ZT] - 0.45) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_YT_ZB] - 0.20) < DBL_EPSILON);
    assert(fabs(ws.s[WALL_SLIP_EDGE_YT_ZT] - 0.25) < DBL_EPSILON);
  }

  return valid;
}

/*****************************************************************************
 *
 *  test_wall_link_normal
 * 
 *  This is independent of the LB model.
 *
 *****************************************************************************/

__host__ int test_wall_link_normal(pe_t * pe, cs_t * cs) {

  map_t * map = NULL;
  wall_t * wall = NULL;
  wall_param_t param = {0};

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  map_create(pe, cs, 1, &map);
  wall_create(pe, cs, map, lb, &wall);

  /* This will probe all possible normal directions for the model */
  param.iswall = 1;
  param.isboundary[X] = 1;
  param.isboundary[Y] = 1;
  param.isboundary[Z] = 1;
  wall_commit(wall, &param);
  
  for (int n = 0; n < wall->nlink; n++) {

    int wn[3] = {0};
    int p = wall->linkp[n];
    int modcv;

    wall_link_normal(wall, n, wn);
    modcv = lb->model.cv[p][X]*lb->model.cv[p][X]
          + lb->model.cv[p][Y]*lb->model.cv[p][Y]
          + lb->model.cv[p][Z]*lb->model.cv[p][Z];

    switch (modcv) {
    case 1:
      /* A link with |cv| = 1 is always normal to the face */
      assert(wn[X] == -lb->model.cv[p][X]);
      assert(wn[Y] == -lb->model.cv[p][Y]);
      assert(wn[Z] == -lb->model.cv[p][Z]);
      break;
    case 2:
      /* A link with |cv| = 2 may be face or edge  */
      if (wn[X] != 0) assert(wn[X] == -lb->model.cv[p][X]);
      if (wn[Y] != 0) assert(wn[Y] == -lb->model.cv[p][Y]);
      if (wn[Z] != 0) assert(wn[Z] == -lb->model.cv[p][Z]);
      break;
    case 3:
      /* A link with |cv| = 3 may be face, edge, or corner */
      if (wn[X] != 0) assert(wn[X] == -lb->model.cv[p][X]);
      if (wn[Y] != 0) assert(wn[Y] == -lb->model.cv[p][Y]);
      if (wn[Z] != 0) assert(wn[Z] == -lb->model.cv[p][Z]);
      break;
    default:
      assert(0);
    }
  }

  wall_free(wall);
  map_free(map);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_link_slip_direction
 *
 *****************************************************************************/

__host__ int test_wall_link_slip_direction(pe_t * pe, cs_t * cs) {

  map_t * map = NULL;
  wall_t * wall = NULL;
  wall_param_t param = {0};

  lb_data_options_t options =  lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  map_create(pe, cs, 1, &map);
  wall_create(pe, cs, map, lb, &wall);

  /* This will probe all possible normal directions for the model */
  param.iswall = 1;
  param.isboundary[X] = 1;
  param.isboundary[Y] = 1;
  param.isboundary[Z] = 1;
  wall_commit(wall, &param);
  
  for (int n = 0; n < wall->nlink; n++) {

    int p = wall->linkp[n];
    int q;
    int wn[3] = {0};

    q = wall_link_slip_direction(wall, n);
    assert(0 < q && q < lb->model.nvel);

    /* This test is slightly circular, but at least consistent... */
    wall_link_normal(wall, n, wn);

    assert((lb->model.cv[p][X] + lb->model.cv[q][X]) == -2*wn[X]);
    assert((lb->model.cv[p][Y] + lb->model.cv[q][Y]) == -2*wn[Y]);
    assert((lb->model.cv[p][Z] + lb->model.cv[q][Z]) == -2*wn[Z]);

    wn[X] += (lb->model.cv[p][X] + lb->model.cv[q][X])/2;
    wn[Y] += (lb->model.cv[p][Y] + lb->model.cv[q][Y])/2;
    wn[Z] += (lb->model.cv[p][Z] + lb->model.cv[q][Z])/2;
    assert(wn[X] == 0);
    assert(wn[Y] == 0);
    assert(wn[Z] == 0);
  }

  wall_free(wall);
  map_free(map);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_link_slip
 *
 *****************************************************************************/

__host__ int test_wall_link_slip(pe_t * pe, cs_t * cs) {

  map_t * map = NULL;
  wall_t * wall = NULL;
  wall_param_t param = {0};

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  map_create(pe, cs, 1, &map);
  wall_create(pe, cs, map, lb, &wall);

  /* This will probe all possible normal directions for the model */
  param.iswall = 1;
  param.isboundary[X] = 1;
  param.isboundary[Y] = 1;
  param.isboundary[Z] = 1;
  wall_commit(wall, &param);
  
  for (int n = 0; n < wall->nlink; n++) {

    int s = wall_link_slip(wall, n);
    int p = wall->linkp[n];
      
    assert(WALL_NO_SLIP <= s && s < WALL_SLIP_MAX);

    /* Make sure these are at least consistent */
    if (s == WALL_SLIP_XBOT) assert(lb->model.cv[p][X] == -1);
    if (s == WALL_SLIP_XTOP) assert(lb->model.cv[p][X] == +1);
    if (s == WALL_SLIP_YBOT) assert(lb->model.cv[p][Y] == -1);
    if (s == WALL_SLIP_YTOP) assert(lb->model.cv[p][Y] == +1);
    if (s == WALL_SLIP_ZBOT) assert(lb->model.cv[p][Z] == -1);
    if (s == WALL_SLIP_ZTOP) assert(lb->model.cv[p][Z] == +1);
    if (p == 0) assert(0);
  }

  wall_free(wall);
  map_free(map);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_commit1
 *
 *****************************************************************************/

__host__ int test_wall_commit1(pe_t * pe, cs_t * cs) {

  map_t * map = NULL;
  wall_t * wall = NULL;
  wall_param_t param = {0};

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  map_create(pe, cs, 1, &map);
  wall_create(pe, cs, map, lb, &wall);

  /* This will probe all possible normal directions for the model */
  param.iswall = 1;
  param.isboundary[X] = 1;
  param.isboundary[Y] = 1;
  param.isboundary[Z] = 1;
  wall_commit(wall, &param);

  assert(wall->nlink > 0);
  assert(wall->linki != NULL);
  assert(wall->linkj != NULL);
  assert(wall->linkp != NULL);
  assert(wall->linku != NULL);

  assert(wall->param->slip.active == 0);
  assert(wall->linkk == NULL);
  assert(wall->linkq == NULL);
  assert(wall->links == NULL);

  wall_free(wall);
  map_free(map);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_commit2
 *
 *****************************************************************************/

__host__ int test_wall_commit2(pe_t * pe, cs_t * cs) {

  map_t * map = NULL;
  wall_t * wall = NULL;
  wall_param_t param = {0};

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  map_create(pe, cs, 1, &map);
  wall_create(pe, cs, map, lb, &wall);

  /* This will probe all possible normal directions for the model */
  param.iswall = 1;
  param.isboundary[X] = 1;
  param.isboundary[Y] = 1;
  param.isboundary[Z] = 1;
  {
    double sbot[3] = {0.5, 0.0, 0.0};
    double stop[3] = {0.0, 0.5, 0.0};
    param.slip = wall_slip(sbot, stop);
  }
  wall_commit(wall, &param);

  assert(wall->nlink > 0);

  assert(wall->param->slip.active);
  assert(wall->linkk != NULL);
  assert(wall->linkq != NULL);
  assert(wall->links != NULL);

  wall_free(wall);
  map_free(map);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_wall_lubr_drag
 *
 *****************************************************************************/

__host__ int test_wall_lubr_drag(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* h > hc => zeta = 0. */
    double eta = 0.1;
    double ah  = 1.25;
    double h   = 1.2;
    double hc  = 1.0;
    double zeta = wall_lubr_drag(eta, ah, h, hc);

    if (fabs(zeta) >= DBL_EPSILON) ifail = 1;
    assert(ifail == 0);
  }
  {
    /* h < hc */
    double eta = 0.1;
    double ah  = 1.25;
    double h   = 1.0;
    double hc  = 1.2;
    double zeta = wall_lubr_drag(eta, ah, h, hc);
    if (fabs(zeta - -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hc)) >= DBL_EPSILON) {
      ifail = -1;
    }
    assert(ifail == 0);
  }

  return ifail;
}
