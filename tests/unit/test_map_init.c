/*****************************************************************************
 *
 *  test_map_init.c
 *
 *  Test of map initialisations is relevant.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Comuting Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "map_init.h"
#include "util.h"

int test_map_init_status_wall(pe_t * pe);
int test_map_init_status_circle(pe_t * pe);
int test_map_init_status_wall_x(pe_t * pe, cs_t * cs);
int test_map_init_status_wall_y(pe_t * pe, cs_t * cs);
int test_map_init_status_wall_z(pe_t * pe, cs_t * cs);
int test_map_init_status_circle_odd(pe_t * pe);
int test_map_init_status_circle_even(pe_t * pe);
int test_map_init_status_body_centred_cubic(pe_t * pe);
int test_map_init_status_face_centred_cubic(pe_t * pe);
int test_map_init_status_simple_cubic(pe_t * pe);

/*****************************************************************************
 *
 *  test_map_init_suite
 *
 *****************************************************************************/

int test_map_init_suite(void) {

  pe_t * pe  = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_map_init_status_wall(pe);
  test_map_init_status_circle(pe);
  test_map_init_status_body_centred_cubic(pe);
  test_map_init_status_face_centred_cubic(pe);
  test_map_init_status_simple_cubic(pe);

  pe_info(pe, "PASS     ./unit/test_map_rt\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_wall
 *
 *****************************************************************************/

int test_map_init_status_wall(pe_t * pe) {

  int ntotal[3] = {16, 8, 4}; /* Something not symmetric in coordinates */
  cs_t * cs   = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  test_map_init_status_wall_x(pe, cs);
  test_map_init_status_wall_y(pe, cs);
  test_map_init_status_wall_z(pe, cs);

  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_wall_x
 *
 *****************************************************************************/

int test_map_init_status_wall_x(pe_t * pe, cs_t * cs) {

  int ntotal[3]  = {0};
  int noffset[3] = {0};

  assert(pe);
  assert(cs);

  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);

  {
    map_t * map = NULL;
    int ndata = 0;

    map_create(pe, cs, ndata, &map);
    map_init_status_wall(map, X);

    {
      /* A small sample of boundary points */
      int status = MAP_STATUS_MAX;
      int index = cs_index(cs, 1, 2, 2);
      map_status(map, index, &status);
      if (noffset[X] == 0) assert(status == MAP_BOUNDARY);
    }

    {
      /* Global volume boundary ... */
      int vol = 0;
      map_volume_allreduce(map, MAP_BOUNDARY, &vol);
      assert(vol == 2*ntotal[Y]*ntotal[Z]);
    }

    {
      /* Fluid sample (all ranks; nlocal[X] > 2) ... */
      int status = MAP_STATUS_MAX;
      int index = cs_index(cs, 2, 1, 1);
      map_status(map, index, &status);
      assert(status == MAP_FLUID);
    }

    {
      /* Fluid volume ... */
      int vol = 0;
      map_volume_allreduce(map, MAP_FLUID, &vol);
      assert(vol == (ntotal[X]-2)*ntotal[Y]*ntotal[Z]);
    }

    map_free(map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_wall_y
 *
 *****************************************************************************/

int test_map_init_status_wall_y(pe_t * pe, cs_t * cs) {

  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int noffset[3] = {0};

  assert(pe);
  assert(cs);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  {
    map_t * map = NULL;
    int ndata = 0;

    map_create(pe, cs, ndata, &map);
    map_init_status_wall(map, Y);

    {
      /* A small sample of boundary points ... */
      int status = MAP_STATUS_MAX;
      int index = cs_index(cs, 2, 1, 2);
      map_status(map, index, &status);
      if (noffset[Y] == 0) assert(status == MAP_BOUNDARY);
    }

    {
      /* Global volume boundary ... */
      int vol = 0;
      map_volume_allreduce(map, MAP_BOUNDARY, &vol);
      assert(vol == ntotal[X]*2*ntotal[Z]);
    }

    map_free(map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_wall_z
 *
 *****************************************************************************/

int test_map_init_status_wall_z(pe_t * pe, cs_t * cs) {

  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int noffset[3] = {0};

  assert(pe);
  assert(cs);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  {
    map_t * map = NULL;
    int ndata = 0;

    map_create(pe, cs, ndata, &map);
    map_init_status_wall(map, Z);

    {
      /* A small sample of boundary points ... */
      int status = MAP_STATUS_MAX;
      int index = cs_index(cs, 2, 2, 1);
      map_status(map, index, &status);
      if (noffset[Z] == 0) assert(status == MAP_BOUNDARY);
    }

    {
      /* Global volume boundary ... */
      int vol = 0;
      map_volume_allreduce(map, MAP_BOUNDARY, &vol);
      assert(vol == ntotal[X]*ntotal[Y]*2);
    }

    map_free(map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_circle
 *
 *****************************************************************************/

int test_map_init_status_circle(pe_t * pe) {

  test_map_init_status_circle_odd(pe);
  test_map_init_status_circle_even(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_circle_odd
 *
 *****************************************************************************/

int test_map_init_status_circle_odd(pe_t * pe) {

  int ntotal[3] = {19, 19, 1}; /* System size: odd numbers */
  cs_t * cs = NULL;
  map_t * map = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  map_create(pe, cs, 0, &map);
  map_init_status_circle_xy(map);

  {
    /* All the edges at least should be solid (no 'leaks') */

    int noffset[3] = {0};
    int nlocal[3] = {0};
    int status1 = -1;
    int status2 = -1;

    cs_nlocal(cs, nlocal);
    cs_nlocal_offset(cs, noffset);

    /* 1st: x = 1, x = Lx for all local y */
    /* 2nd: y = 1, y = Ly for all local x */

    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int i1 = cs_index(cs,         1, jc, 1);
      int i2 = cs_index(cs, nlocal[X], jc, 1);
      map_status(map, i1, &status1);
      map_status(map, i2, &status2);
      if (noffset[X] == 0)                     assert(status1 == MAP_BOUNDARY);
      if (noffset[X] + nlocal[X] == ntotal[X]) assert(status2 == MAP_BOUNDARY);
    }

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      int j1 = cs_index(cs, ic,         1, 1);
      int j2 = cs_index(cs, ic, nlocal[Y], 1);
      map_status(map, j1, &status1);
      map_status(map, j2, &status2);
      if (noffset[Y] == 0)                     assert(status1 == MAP_BOUNDARY);
      if (noffset[Y] + nlocal[Y] == ntotal[Y]) assert(status2 == MAP_BOUNDARY);
    }
  }

  {
    /* Global volumes ... */
    int nsolid = -1;
    int nfluid = -1;
    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    map_volume_allreduce(map, MAP_FLUID,    &nfluid);

    /* Reference answers are historical, by inspection */
    assert(nsolid == 136);
    assert(nfluid == 225);
    assert((nfluid + nsolid) == ntotal[X]*ntotal[Y]);
  }

  map_free(map);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 * test_map_init_status_circle_even
 *
 *****************************************************************************/

int test_map_init_status_circle_even(pe_t * pe) {

  int ntotal[3] = {18, 18, 2}; /* System size: even numbers */
  cs_t * cs = NULL;
  map_t * map = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  map_create(pe, cs, 0, &map);
  map_init_status_circle_xy(map);
  {
    /* Just check the global volumes ... */
    int nsolid = -1;
    int nfluid = -1;
    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    map_volume_allreduce(map, MAP_FLUID,    &nfluid);

    /* Reference answers are historical, by inspection */
    assert(nsolid == 232);
    assert(nfluid == 416);
    assert((nfluid + nsolid) == ntotal[X]*ntotal[Y]*ntotal[Z]);
  }

  map_free(map);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_init_status_body_centred_cubic
 *
 *  The packing fraction is pi sqrt(3)/8.
 *  E.g., https://en.wikipedia.org/wiki/Atomic_packing_factor
 *
 *  These three function are looking rather similar, but one could
 *  actually put in some more specific tests for each one.
 *
 *****************************************************************************/

int test_map_init_status_body_centred_cubic(pe_t * pe) {

  int ierr = 0;

  cs_t * cs = NULL;
  map_t * map = NULL;
  int ndata = 0;

  /* Recall we must have system size a multiple of the lattice constant */
  int acell = 20;
  int ntotal[3] = {acell, acell, acell};

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  map_create(pe, cs, ndata, &map);
  map_init_status_body_centred_cubic(map, acell);

  {
    /* Compute solid fraction */
    int nsolid = 0;
    double sf = 0.0;
    PI_DOUBLE(pi);

    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    sf = 1.0*nsolid/(ntotal[X]*ntotal[Y]*ntotal[Z]);

    /* Should be good to 0.01 */
    assert(fabs(sf - pi*sqrt(3.0)/8.0) < 0.01);
    if (fabs(sf - pi*sqrt(3.0)/8.0) >= 0.01) ierr = -1;
  }

  map_free(map);
  cs_free(cs);

  return ierr;
}

/*****************************************************************************
 *
 *  test_map_init_status_face_centred_cubic
 *
 *  The packing fraction should be pi sqrt(2)/6
 *
 *****************************************************************************/

int test_map_init_status_face_centred_cubic(pe_t * pe) {

  int ierr = 0;

  cs_t * cs = NULL;
  map_t * map = NULL;
  int ndata = 0;

  /* Recall we must have system size a multiple of the lattice constant */
  int acell = 20;
  int ntotal[3] = {acell, acell, acell};

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  map_create(pe, cs, ndata, &map);
  map_init_status_face_centred_cubic(map, acell);

  {
    /* Compute solid fraction */
    int nsolid = 0;
    double sf = 0.0;
    PI_DOUBLE(pi);

    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    sf = 1.0*nsolid/(ntotal[X]*ntotal[Y]*ntotal[Z]);

    /* Should be good to 0.01 */
    assert(fabs(sf - pi*sqrt(2.0)/6.0) < 0.01);
    if (fabs(sf - pi*sqrt(2.0)/6.0) >= 0.01) ierr = -1;
  }

  map_free(map);
  cs_free(cs);

  return ierr;
}

/*****************************************************************************
 *
 *  test_map_init_status_simple_cubic
 *
 *  The packing fraction should be pi/6.
 *
 *****************************************************************************/

int test_map_init_status_simple_cubic(pe_t * pe) {

  int ierr = 0;

  cs_t * cs = NULL;
  map_t * map = NULL;
  int ndata = 0;

  /* Recall we must have system size a multiple of the lattice constant */
  int acell = 20;
  int ntotal[3] = {acell, acell, acell};

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  map_create(pe, cs, ndata, &map);
  map_init_status_simple_cubic(map, acell);

  {
    /* Compute solid fraction */
    int nsolid = 0;
    double sf = 0.0;
    PI_DOUBLE(pi);

    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    sf = 1.0*nsolid/(ntotal[X]*ntotal[Y]*ntotal[Z]);
    /* Should be good to 0.01 */
    assert(fabs(sf - pi/6.0) < 0.01);
    if (fabs(sf - pi/6.0) >= 0.01) ierr = -1;
  }

  map_free(map);
  cs_free(cs);

  return ierr;
}
