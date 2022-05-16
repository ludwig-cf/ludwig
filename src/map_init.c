/*****************************************************************************
 *
 *  map_init.h
 *
 *  Various map status/data initialisations.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022
 *
 *  Contributing authors;
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Jurij Sablic provided the crystal sphere packing routines
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "map_init.h"


/*****************************************************************************
 *
 *  map_init_status_circle_xy
 *
 *  Centre at (Lx/2, Ly/2) with solid boundary at L = 1 and L = L in (x,y).
 *  We could insist that the system is square.
 *
 *****************************************************************************/

int map_init_status_circle_xy(map_t * map) {

  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int noffset[3] = {0};
  double x0, y0, r0;

  cs_t * cs = NULL;

  assert(map);

  cs = map->cs;
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  /* Check (x,y) is square and assign a centre and radius */

  if (ntotal[X] != ntotal[Y]) {
    pe_t * pe = map->pe;
    pe_fatal(pe, "map_init_status_circle_xy must have Lx == Ly\n");
    /* Could move that to higher level and return a failure. */
  }

  x0 = 0.5*(1 + ntotal[X]); /* ok for even, odd ntotal[X] */
  y0 = 0.5*(1 + ntotal[Y]);
  r0 = 0.5*(ntotal[X] - 2);

  /* Assign status */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    double x = (noffset[X] + ic) - x0;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      double y = (noffset[Y] + jc) - y0;

      double r = x*x + y*y;
      char status = MAP_BOUNDARY;

      if (r <= r0*r0) status = MAP_FLUID;

      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, status);
      }
    }
  }

  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_wall
 *
 *  Initialise boundary at L = 1 and L = L in given direction.
 *  Note we do not set any fluid sites.
 *
 *****************************************************************************/

__host__ int map_init_status_wall(map_t * map, int id) {

  int ntotal[3]  = {0};
  int nlocal[3]  = {0};
  int noffset[3] = {0};
  cs_t * cs = NULL;

  assert(id == X || id == Y || id == Z);
  assert(map);

  cs = map->cs;
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int ix = noffset[X] + ic;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int iy = noffset[Y] + jc;
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int iz = noffset[Z] + kc;
	int index = cs_index(cs, ic, jc, kc);

	if (id == X && (ix == 1 || ix == ntotal[X])) { 
	  map_status_set(map, index, MAP_BOUNDARY);
	}
	if (id == Y && (iy == 1 || iy == ntotal[Y])) { 
	  map_status_set(map, index, MAP_BOUNDARY);
	}
	if (id == Z && (iz == 1 || iz == ntotal[Z])) { 
	  map_status_set(map, index, MAP_BOUNDARY);
	}

      }
    }
  }

  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_simple_cubic
 *
 *  The cell size is an integer which must divide exactly (Lx, Ly, Lz).
 *
 *****************************************************************************/

int map_init_status_simple_cubic(map_t * map, int acell) {

  cs_t * cs = NULL;
  int nlocal[3] = {0};
  int noffset[3] = {0};

  assert(map);

  cs = map->cs;
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int ix = noffset[X] + ic - 1;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int iy = noffset[Y] + jc - 1;
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int iz = noffset[Z] + kc - 1;
	int index = cs_index(cs, ic, jc, kc);

	/* distance between the node (i,j,k) and the centre of the
	   nearest crystalline particle, located at the edges of
	   the crystalline cell */

	double dx = ix - round(1.0*ix/acell)*acell;
	double dy = iy - round(1.0*iy/acell)*acell;
	double dz = iz - round(1.0*iz/acell)*acell;

	double radius = 0.5*acell;
	double r = sqrt(dx*dx + dy*dy + dz*dz);

	int status = MAP_FLUID;
	if (r <= radius) status = MAP_BOUNDARY;
	map_status_set(map, index, status);
      }
    }
  }

  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_body_centred_cubic
 *
 *****************************************************************************/

int map_init_status_body_centred_cubic(map_t * map, int acell) {

  cs_t * cs = NULL;
  int nlocal[3] = {0};
  int noffset[3] = {0};
  double radius;

  assert(map);

  cs = map->cs;
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  /* Here is a radius that will fit ... */
  radius = 0.25*sqrt(3.0)*acell;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int ix = noffset[X] + ic - 1;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int iy = noffset[Y] + jc - 1;
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int iz = noffset[Z] + kc - 1;
	int index = cs_index(cs, ic, jc, kc);

	/* distance between (ix,iy,iz) and the centre of the nearest
	 * particle, located at the edge of the crystalline cell */

	double dx = ix - round(1.0*ix/acell)*acell;
	double dy = iy - round(1.0*iy/acell)*acell;
	double dz = iz - round(1.0*iz/acell)*acell;

	double r = sqrt(dx*dx + dy*dy + dz*dz);

	int status = MAP_FLUID;
	if (r <= radius) status = MAP_BOUNDARY;

	/* distance between (ix,iy,iz) and the centre of the particle
	 * located at the centre of the crystalline cell */

	dx = ix - (floor(1.0*ix/acell) + 0.5)*acell;
	dy = iy - (floor(1.0*iy/acell) + 0.5)*acell;
	dz = iz - (floor(1.0*iz/acell) + 0.5)*acell;

	r = sqrt(dx*dx + dy*dy + dz*dz);
	if (r <= radius) status = MAP_BOUNDARY;

	map_status_set(map, index, status);
      }
    }
  }

  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_face_centred_cubic
 *
 *  The lattice constant acell must divide evenly all the
 *  lengths of the system (lx,ly,lz) or the results will
 *  be unpredictable.
 *
 *****************************************************************************/

int map_init_status_face_centred_cubic(map_t * map, int acell) {

  cs_t * cs = NULL;
  int nlocal[3] = {0};
  int noffset[3] = {0};
  double radius;

  assert(map);

  cs = map->cs;
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  radius = 0.25*sqrt(2.0)*acell;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int ix = noffset[X] + ic - 1;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int iy = noffset[Y] + jc - 1;
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int iz = noffset[Z] + kc - 1;
	int index = cs_index(cs, ic, jc, kc);
	int status = MAP_FLUID;

	/* particle located at the edge of the crystalline cell */

	double dx = ix - round(1.0*ix/acell)*acell;
	double dy = iy - round(1.0*iy/acell)*acell;
	double dz = iz - round(1.0*iz/acell)*acell;

	double r = sqrt(dx*dx + dy*dy + dz*dz);
	if (r <= radius) status = MAP_BOUNDARY;

	/* particle located at the centre of the xy-surface */

	dx = ix - (floor(1.0*ix/acell) + 0.5)*acell;
	dy = iy - (floor(1.0*iy/acell) + 0.5)*acell;
	dz = iz - (round(1.0*iz/acell)      )*acell;

	r = sqrt(dx*dx + dy*dy + dz*dz);
	if (r <= radius) status = MAP_BOUNDARY;

	/* particle located at the centre of the xz-surface */

	dx = ix - (floor(1.0*ix/acell) + 0.5)*acell;
	dy = iy - (round(1.0*iy/acell)      )*acell;
	dz = iz - (floor(1.0*iz/acell) + 0.5)*acell;

	r = sqrt(dx*dx+ dy*dy + dz*dz);
	if (r <= radius) status = MAP_BOUNDARY;

	/* sphere located at the centre of the yz-surface */

	dx = ix - (round(1.0*ix/acell)      )*acell;
	dy = iy - (floor(1.0*iy/acell) + 0.5)*acell;
	dz = iz - (floor(1.0*iz/acell) + 0.5)*acell;

	r = sqrt(dx*dx + dy*dy + dz*dz);
	if (r <= radius) status = MAP_BOUNDARY;

	map_status_set(map, index, status);
      }
    }
  }

  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_print_section
 *
 *  Just for fun really; serial only makes sense.
 *
 *  The id argument is the direction not to include;
 *  i.e., X for YZ section etc.
 *  The ord argument is the value of the coordinate in the
 *  relevant direction e.g., ic = 1 for pid = X.
 *
 *****************************************************************************/

int map_init_status_print_section(map_t * map, int id, int ord) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  int nlocal[3] = {0};
  int status = -1;

  assert(map);
  assert(id == X || id == Y || id == Z);

  pe = map->pe;
  cs = map->cs;
  cs_nlocal(cs, nlocal);

  pe_info(pe, "\nCross section (%d = fluid, %d = solid)\n", 0, 1);

  switch (id) {
  case X:
    for (int jc = 1; jc <= nlocal[Y]; jc++ ) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ord, jc, kc);
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY)  pe_info(pe, " %d", 1);
	if (status == MAP_FLUID)     pe_info(pe, " %d", 0);
      }
      printf("\n");
    }
    break;
  case Y:
    for (int ic = 1; ic <= nlocal[X]; ic++ ) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, ord, kc);
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY)  pe_info(pe, " %d", 1);
	if (status == MAP_FLUID)     pe_info(pe, " %d", 0);
      }
      printf("\n");
    }
    break;
  case Z:
    for (int ic = 1; ic <= nlocal[X]; ic++ ) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	int index = cs_index(cs, ic, jc, ord);
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY)  pe_info(pe, " %d", 1);
	if (status == MAP_FLUID)     pe_info(pe, " %d", 0);
      }
      printf("\n");
    }
    break;
  default:
    pe_fatal(pe, "INTERNAL ERROR: Bad id in map_status_section\n");
  }

  {
    int nsolid = -1;
    int nfluid = -1;
    int ntotal = nlocal[X]*nlocal[Y]*nlocal[Z]; /* Serial only */

    map_volume_allreduce(map, MAP_BOUNDARY, &nsolid);
    map_volume_allreduce(map, MAP_FLUID,    &nfluid);

    pe_info(pe, "ntotal = %d nsolid = %d nfluid = %d nsolid fraction: %f \n",
	    ntotal, nsolid, nfluid, (1.0*nsolid)/ntotal);
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_data_uniform
 *
 *  Initialise data (typically wetting constants) to given values for
 *  sites of given target status.
 *
 *****************************************************************************/

int map_init_data_uniform(map_t * map, int target, double * data) {

  int status = -1;

  assert(map);
  assert(data);

  {
    cs_t * cs = map->cs;
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  map_status(map, index, &status);
	  if (status == target) map_data_set(map, index, data);
	}
      }
    }
  }

  return 0;
}
