/*****************************************************************************
 *
 *  capillary.c
 *
 *  This utility produces an output file suitable for initialising
 *  a capillary structure in Ludwig. 
 *
 *  Some examples of uniform, and less uniform capillary initialisations.
 *  Options are selected at compile time (below) at the momoent.
 *
 *  The output should be capillary.dat      [for human consumption]
 *                       capillary.001-001  [for initial input to run]
 *
 *  (c) 2008-2021 The University of Edinburgh
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "map_init.h"
#include "symmetric.h"
#include "fe_ternary.h"

/* SYSTEM SIZE */
/* Set the system size as desired. Clearly, this must match the system
 * set in the main input file for Ludwig. */

const int xmax = 20;
const int ymax = 20;
const int zmax = 20;

const int crystalline_cell_size = 10; /* Must devide all lengths */

/* CROSS SECTION */
/* You can choose a square or circular cross section */

enum {CIRCLE, SQUARE, XWALL, YWALL, ZWALL, XWALL_OBSTACLES, XWALL_BOTTOM,
      SPECIAL_CROSS, SIMPLE_CUBIC, BODY_CENTRED_CUBIC, FACE_CENTRED_CUBIC};

const int xsection = XWALL_OBSTACLES;

/* "Obstacles": Modify the local geometry of the wall */

int obstacle_number = 1; /* number of obstacles per wall */
int obstacle_length = 6; /* along the wall direction */
int obstacle_height = 8; /* perpendicular from wall */
int obstacle_depth  = 6; /* perpendicular to length and height */
			 /* NOTE: obstacle_depth == xmax/ymax/zmax 
				  means obstacles don't have a z-boundary */

/* SURFACE CHARGE */

const double sigma = 0.125;

/* OUTPUT */
/* You can generate a file with solid/fluid status information only,
 * or one which includes the wetting parameters or charge Q. */

enum {STATUS_ONLY, STATUS_WITH_C_H, STATUS_WITH_SIGMA,
      STATUS_WITH_H1_H2};
const int output_type = STATUS_WITH_SIGMA;

int map_special_cross(map_t * map);

int map_xwall_obstacles(map_t * map, double sigma);
int capillary_write_ascii_serial(pe_t * pe, cs_t * cs, map_t * map);

/*****************************************************************************
 *
 *  main program
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  int ndata = 0;
  double data[2];  /* ndata = 2 max at the moment. for uniform cases */
  map_t * map = NULL;

  int k_pic = 0;   /* k-value section to screen (k_pic = 0 is no picture) */


  MPI_Init(&argc, &argv);
  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  cs_create(pe, &cs);
  {
    int ntotal[3] = {xmax, ymax, zmax};
    cs_ntotal_set(cs, ntotal);
  }
  cs_init(cs);

  switch (output_type) {
  case STATUS_WITH_C_H:

    {
      /* SYMMETRIC FREE ENERGY */
      /* Some default values for free energy: */
      fe_symm_param_t fe = {.a =    -0.0625,
			    .b =     0.0625,
			    .kappa = 0.053,
			    .c     = 0.0,
			    .h     = 0.0};
      double h, h1, theta;

      printf("Free energy parameters:\n");
      printf("free energy parameter kappa = %f\n", fe.kappa);
      printf("free energy parameter B     = %f\n", fe.b);
      printf("surface free energy   H     = %f\n", fe.h);
      h = fe.h*sqrt(1.0/(fe.kappa*fe.b));
      printf("dimensionless parameter h   = %f\n", h);
      h1 = 0.5*(-pow(1.0 - h, 1.5) + pow(1.0 + h, 1.5));
      printf("dimensionless parameter h1=cos(theta)   = %f\n", h1);
      theta = acos(h1);
      printf("contact angle theta         = %f radians\n", theta);
      theta = theta*180.0/(4.0*atan(1.0));
      printf("                            = %f degrees\n", theta);

      /* We must have this order "C, H" ... */
      ndata = 2;
      data[0] = fe.c;
      data[1] = fe.h;
    }

    break;

  case STATUS_WITH_SIGMA:
    /* Just a surface charge... */
    ndata = 1;
    data[0] = sigma;
    printf("Surface charge sigma = %f\n", sigma);
    break;

  case STATUS_WITH_H1_H2:

    {

      /* TERNARY FREE ENERGY PARAMETERS */
      /* Set the fluid and solid free energy parameters. The fluid parameters
       * must match those used in the main calculation. See Semprebon et al.
       * Phys. Rev. E (2016) for details. */

      /* Note on wetting: there is no independent h3 in the current picture. */

      fe_ternary_param_t fe = { .alpha   =  1.0,
			        .kappa1  =  0.012,
			        .kappa2  =  0.05,
			        .kappa3  =  0.05,
			        .h1      =  0.0007,
			        .h2      = -0.005
      };

      double f1,f2,f3,cos_theta12,cos_theta23,cos_theta31;
      double theta12,theta23,theta31;

      /* Constraint for h3: */
      fe.h3 = fe.kappa3*(-(fe.h1/fe.kappa1) - (fe.h2/fe.kappa2));

      printf("Ternary free energy parameters:\n");
      printf("free energy parameter kappa1 = %f\n", fe.kappa1);
      printf("free energy parameter kappa2 = %f\n", fe.kappa2);
      printf("free energy parameter kappa3 = %f\n", fe.kappa3);
      printf("free energy parameter alpha = %f\n",  fe.alpha);

    f1=pow(fe.alpha*fe.kappa1+4*fe.h1,1.5)-pow(fe.alpha*fe.kappa1-4*fe.h1,1.5);
    f1=f1/sqrt(fe.alpha*fe.kappa1);
    f2=pow(fe.alpha*fe.kappa2+4*fe.h2,1.5)-pow(fe.alpha*fe.kappa2-4*fe.h2,1.5);
    f2=f2/sqrt(fe.alpha*fe.kappa2);
    f3=pow(fe.alpha*fe.kappa3+4*fe.h3,1.5)-pow(fe.alpha*fe.kappa3-4*fe.h3,1.5);
    f3=f3/sqrt(fe.alpha*fe.kappa3);

    cos_theta12=f1/(2.0*(fe.kappa1+fe.kappa2))-f2/(2.0*(fe.kappa1+fe.kappa2));
    cos_theta23=f2/(2.0*(fe.kappa2+fe.kappa3))-f3/(2.0*(fe.kappa2+fe.kappa3));
    cos_theta31=f3/(2.0*(fe.kappa3+fe.kappa1))-f1/(2.0*(fe.kappa3+fe.kappa1));
    printf("dimensionless parameters cos(theta12)   = %f\n", cos_theta12);
    printf("dimensionless parameters cos(theta23)   = %f\n", cos_theta23);
    printf("dimensionless parameters cos(theta13)   = %f\n", cos_theta31);
    theta12=acos(cos_theta12);
    theta23=acos(cos_theta23);
    theta31=acos(cos_theta31);
    printf("contact angle theta12         = %f radians\n", theta12);
    theta12=theta12*180.0/(4.0*atan(1.0));
    printf("contact angle theta12         = %f degrees\n", theta12);
    printf("contact angle theta23         = %f radians\n", theta23);
    theta23=theta23*180.0/(4.0*atan(1.0));
    printf("contact angle theta23         = %f degrees\n", theta23);
    printf("contact angle theta31         = %f radians\n", theta31);
    theta31=theta31*180.0/(4.0*atan(1.0));
    printf("contact angle theta31         = %f degrees\n", theta31);

    /* We should have this order for additional data: h1, h2 */
    /* No h3 is required. */
    ndata = 2;
    data[0] = fe.h1;
    data[1] = fe.h2;
    }
    break;

  default:
    ndata = 0;
  }

  /* Now we know ndata, allocate a map structure... */

  map_create(pe, cs, ndata, &map);

  /* Structures */

  switch (xsection) {

  case SPECIAL_CROSS:
    map_special_cross(map);
    k_pic = 1;
    break;

  case CIRCLE:
    map_init_status_circle_xy(map);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    /* Case of z1_h, z2_h is special and needs to be recovered */
    k_pic = 1;
    break;

  case SIMPLE_CUBIC:
    /* simple cubic crystal */
    map_init_status_simple_cubic(map, crystalline_cell_size);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    k_pic = 1;
    break;

  case BODY_CENTRED_CUBIC:
    /* body centered cubic crystal */
    map_init_status_body_centred_cubic(map, crystalline_cell_size);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    k_pic = 6; /* For historical purposes */
    break;

  case FACE_CENTRED_CUBIC:
    /* face centered cubic crystal */
    map_init_status_face_centred_cubic(map, crystalline_cell_size);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    k_pic = 1;
    break;

  case SQUARE:
    map_init_status_wall(map, X);
    map_init_status_wall(map, Y);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    /* Case of z1_h and z2_h requires special treatment */
    k_pic = 1;
    break;

  case XWALL:
    map_init_status_wall(map, X);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    break;

  case XWALL_BOTTOM:

    /* Only lower wall x = 1 wets; upper wall defaults to zero/neutral. */

    map_init_status_wall(map, X);

    {
      int nlocal[3] = {};
      int noffset[3] = {};
      cs_nlocal(cs, nlocal);
      cs_nlocal_offset(cs, noffset);
      if (noffset[X] == 0) {
	for (int jc = 1; jc <= nlocal[Y]; jc++) {
	  for (int kc = 1; kc <= nlocal[Z]; kc++) {
	    int index = cs_index(cs, 1, jc, kc);
	    map_data_set(map, index, data);
	  }
	}
      }
    }
    break;

  case XWALL_OBSTACLES:
    pe_info(pe, "Number of obstacles (top):   %d\n", obstacle_number);
    pe_info(pe, "Number of obstacles (bttom): %d\n", obstacle_number);
    pe_info(pe, "Obstacle length:             %d\n", obstacle_length);
    pe_info(pe, "Obstacle height:             %d\n", obstacle_height);
    pe_info(pe, "Obstacle depth:              %d\n", obstacle_depth);
    map_xwall_obstacles(map, sigma);
    k_pic = 10;
    break;

  case YWALL:
    map_init_status_wall(map, Y);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    break;

  case ZWALL:
    map_init_status_wall(map, Z);
    map_init_data_uniform(map, MAP_BOUNDARY, data);
    break;

  default:
    printf("No cross-section!\n");
  }

  /* picture to terminal */

  if (k_pic > 0) map_init_status_print_section(map, Z, k_pic);

  /* Utility output ascii to file */
  capillary_write_ascii_serial(pe, cs, map);

  /* Standard file output */
  {
    int io_grid[3] = {1, 1, 1};
    map_init_io_info(map, io_grid, IO_FORMAT_BINARY, IO_FORMAT_ASCII);
    io_write_data(map->info, "capillary", map);
  }

  map_free(map);
  cs_free(cs);
  pe_free(pe);

  MPI_Finalize();

  return 0;
}

/******************************************************************************
 *
 *  map_special_cross
 *
 *  Make a cross shape which looks like a '+' in the x-y plane.
 *
 *  We use the system size from above so
 *   (Lx, Ly, Lz) = (xmax, ymax, zmax).
 *
 *  Along the x-direction we have a periodic channel of width
 *  W, with the centre of the channel at Ly/2. The channel is
 *  'open', ie., periodic at x = 1 and x = Lx.
 *
 *  In the y-direction we have two dead-end arms of length
 *  W_arm in the x-direction. The extent of each arm in the
 *  y-direction is (Ly - W - 2)/2 (one point is solid at
 *  y = 1 and y = Ly to form the dead-end.
 *
 *  The height is the same everywhere in the z-direction
 *  (height = Lz - 2, ie., it is 'closed' at top and bottom).
 *
 *  Example: xmax = 19, ymax = 13, zmax = 9; w = 5, w_arm = 4.
 *  To minimise computational modes, it's a good idea to have
 *  xmax, ymax, zmax, odd numbers.
 *
 *****************************************************************************/

int map_special_cross(map_t * map) {

  const int w = 5;
  const int w_arm = 4;
  int nlocal[3] = {};
  int x0, x1, j0, j1;

  cs_t * cs = NULL;

  assert(map);

  cs = map->cs;
  cs_nlocal(cs, nlocal);

  /* The plan.
   * 1. Set all the points to solid
   * 2. Drive a period channel along the x-direction
   * 3. Drive a non-periodic channel across the y-direction */

  assert(nlocal[X] == xmax); /* Serial */
  assert(nlocal[Y] == ymax); /* Serial */
  assert(nlocal[Z] == zmax); /* Serial */

  printf("Special cross routine\n");
  printf("Lx Ly Lz: %4d %4d %4d\n", nlocal[X], nlocal[Y], nlocal[Z]);
  printf("Channel width: %3d\n", w);
  printf("Arm length:    %3d\n", w_arm);

  assert(nlocal[Y] % 2);  /* Odd number of points */
  assert(w % 2);          /* Odd number of points */

  /* Set all points to solid */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);
      }
    }
  }

  /* Centred peridoic channel in x-direction */

  j0 = (nlocal[Y]+1)/2 - (w-1)/2;
  j1 = (nlocal[Y]+1)/2 + (w-1)/2;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = j0; jc <= j1; jc++) {
      for (int kc = 2; kc <= nlocal[Z]-1; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_FLUID);
      }
    }
  }

  /* The 'arms' of the cross */

  x0 = (nlocal[X] - w_arm + 1)/2 + (nlocal[X] % 2);
  x1 = x0 + w_arm - 1;

  for (int ic = x0; ic <= x1; ic++) {
    for (int jc = 2; jc <= nlocal[Y]-1; jc++) {
      for (int kc = 2; kc <= nlocal[Z]-1; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_FLUID);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_xwall_obstacles
 *
 *  Must be serial at the moment.
 *
 *****************************************************************************/

int map_xwall_obstacles(map_t * map, double sigma) {

  cs_t * cs = NULL;
  int nlocal[3] = {};

  int obst_start[2*obstacle_number][3];
  int obst_stop[2*obstacle_number][3];

  /* Eliminate global variables here... */
  int nobs = obstacle_number;
  int obsext[3] = {obstacle_height, obstacle_length, obstacle_depth};

  assert(map);

  cs = map->cs;
  cs_nlocal(cs, nlocal);

  /* x-extent (doesn't really matter if this overlaps wall)  */

  for (int iobst = 0; iobst < nobs; iobst++) {
    /* bottom */
    obst_start[iobst][X] = 1;
    obst_stop[iobst][X]  = obst_start[iobst][X] + obsext[X];
    /* top */
    obst_start[nobs+iobst][X] = nlocal[X] - obsext[X];
    obst_stop[nobs+iobst][X]  = nlocal[X] - 1;
  }

  /* y-extent: (iobst % nobs) to be sure same at top and bottom */

  for (int iobst = 0; iobst < 2*nobs; iobst++) {
    double dy = (nlocal[Y] - nobs*obsext[Y])/nobs;
    double y0 = 1 + dy/2 + (iobst % nobs)*(dy + obsext[Y]);
    obst_start[iobst][Y] = y0;
    obst_stop[iobst][Y]  = y0 + obsext[Y] - 1;
  }

  /* z-extent is fixed: all centrally positioned */

  for (int iobst = 0; iobst < 2*nobs; iobst++) {
    obst_start[iobst][Z] = 1 + (nlocal[Z] - obsext[Z])/2;
    obst_stop[iobst][Z]  =     obst_start[iobst][Z] + obsext[Z] - 1;
  }


  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Y]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	int status = MAP_FLUID;

	/* walls, then obstacles */

	if (ic == 1 || ic == nlocal[X]) status = MAP_BOUNDARY;

	for (int iobst = 0; iobst < 2*obstacle_number ; iobst++) {

	  int isi = (obst_start[iobst][X] <= ic && ic <= obst_stop[iobst][X]);
	  int isj = (obst_start[iobst][Y] <= jc && jc <= obst_stop[iobst][Y]);
	  int isk = (obst_start[iobst][Z] <= kc && kc <= obst_stop[iobst][Z]);

	  if (isi && isj && isk) status = MAP_BOUNDARY;
	}

	map_status_set(map, index, status);
      }
    }
  }

  /* Set surface charge. This is only at surfaces (not interior solid).
   * So examine fluid sites, and set if we have solid nearest neighbour... */

  /* halo_status */
  /* Separate routine... */

  assert(map->ndata = 1);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	int status = -1;
	double data[1] = {sigma};
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY) continue;

	/* Look at 6 adjacent sites */
	{
	  int im1 = cs_index(cs, ic-1, jc, kc);
	  map_status(map, im1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, im1, data);
	}
	{
	  int ip1 = cs_index(cs, ic+1, jc, kc);
	  map_status(map, ip1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, ip1, data);
	}
	{
	  int jm1 = cs_index(cs, ic, jc-1, kc);
	  map_status(map, jm1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, jm1, data);
	}
	{
	  int jp1 = cs_index(cs, ic, jc+1, kc);
	  map_status(map, jp1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, jp1, data);
	}
	{
	  int km1 = cs_index(cs, ic, jc, kc-1);
	  map_status(map, km1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, km1, data);
	}
	{
	  int kp1 = cs_index(cs, ic, jc, kc+1);
	  map_status(map, kp1, &status);
	  if (status == MAP_BOUNDARY) map_data_set(map, kp1, data);
	}
	/* next site */
      }
    }
  }

  /* halo data */

  return 0;
}

/*****************************************************************************
 *
 *  capillary_write_ascii_serial
 *
 *****************************************************************************/

int capillary_write_ascii_serial(pe_t * pe, cs_t * cs, map_t * map) {

  const char * filename = "capillary.dat";

  int nlocal[3] = {};
  FILE * fp = NULL;

  assert(pe);
  assert(cs);
  assert(map);

  cs_nlocal(cs, nlocal);

  fp = fopen(filename, "w");
  if (fp == NULL) return -1;

  /* Header comment */
  fprintf(fp, "# ic   jc   kc map  [data..]\n");

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	int status = -1;
	double data[map->ndata];

	map_status(map, index, &status);
	map_data(map, index, data);

	fprintf(fp, "%4d %4d %4d %3d", ic, jc, kc, status);
	for (int nd = 0; nd < map->ndata; nd++) {
	  fprintf(fp, " %22.15e", data[nd]);
	}
	fprintf(fp, "\n");
      }
    }
  }

  fclose(fp);

  return 0;
}
