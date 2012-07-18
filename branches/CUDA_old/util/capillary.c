/*****************************************************************************
 *
 *  capillary.c
 *
 *  Compile and link with -lm for the maths library. 
 *
 *  This utility produces an output file suitable for initialising
 *  a capillary structure in Ludwig. 
 *
 *  It is assumed the periodic dimension is z, and that the system
 *  is square in the x-y directions. No 'leakage' in the x-y
 *  directions is ensured by making the last site in each direction
 *  solid.
 *
 *  The various system parameters should be set at comiple time,
 *  and are described below. The output file is always in BINARY
 *  format.
 *
 *  1. Output capaillary structure
 *  Set the required parameters and invoke with no argument
 *      ./a.out
 *   
 *  2. Profiles
 *  If the program is invoked with a single phi output file
 *  argument, e.g.,
 *      ./a.out phi-001000.001-001
 *  a scatter plot of the profile of the interface in the
 *  wetting half of the capillary will be produced. That
 *  is height vs r, the radial distance from the centre.
 *  The output file should match the capillary structure!
 *
 *  $Id: capillary.c,v 1.3 2008-11-26 19:34:32 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/site_map.h"

/* SYSTEM SIZE */
/* Set the system size as desired. Clearly, this must match the system
 * set in the main input file for Ludwig. */

const int xmax = 10;
const int ymax = 10;
const int zmax = 32;

/* CROSS SECTION */
/* You can choose a square or circular cross section */

enum {CIRCLE, SQUARE};
const int xsection = CIRCLE;

/* FREE ENRGY PARAMETERS */
/* Set the fluid and solid free energy parameters. The fluid parameters
 * must match those used in the main calculation. See Desplat et al.
 * Comp. Phys. Comm. (2001) for details. */

const double kappa = 0.04;
const double B = 0.0625;
const double H = 0.01;

/* WETTING. */
/* A section of capillary between z1 and z2 (inclusive) will have
 * wetting property H = H, the remainder H = 0 */

const int z1 = 1;
const int z2 = 16;

/* OUTPUT */
/* You can generate a file with solid/fluid status information only,
 * or one which includes the wetting parameter H. */

enum {STATUS_ONLY, STATUS_WITH_H};
const int output_type = STATUS_WITH_H;

/* OUTPUT FILENAME */

const char * filename = "capillary.001-001";

static void profile(const char *);

/*****************************************************************************
 *
 *  main program
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  char * map_in;
  FILE * fp_orig;
  int i, j, k, n;
  int nsolid = 0;

  double * map_h;
  double rc = 0.5*(xmax-2);
  double x0 = 0.5*xmax + 0.5;
  double y0 = 0.5*ymax + 0.5;
  double x, y, r;
  double h, h1, theta;

  if (argc == 2) profile(argv[1]);

  printf("Free energy parameters:\n");
  printf("free energy parameter kappa = %f\n", kappa);
  printf("free energy parameter B     = %f\n", B);
  printf("surface free energy   H     = %f\n", H);
  h = H*sqrt(1.0/(kappa*B));
  printf("dimensionless parameter h   = %f\n", h);
  h1 = 0.5*(-pow(1.0 - h, 1.5) + pow(1.0 + h, 1.5));
  theta = acos(h1);
  printf("contact angle theta         = %f radians\n", theta);
  theta = theta*180.0/(4.0*atan(1.0));
  printf("                            = %f degrees\n", theta);

  map_in = (char *) malloc(xmax*ymax*zmax*sizeof(char));
  if (map_in == NULL) exit(-1);

  map_h = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (map_h == NULL) exit(-1);

  if (xsection == CIRCLE) {

    for (i = 0; i < xmax; i++) {
      x = 1.0 + i - x0;
      for (j = 0; j < ymax; j++) {
	y = 1.0 + j - y0;
	for (k = 0; k < zmax; k++) {
	  n = ymax*zmax*i + zmax*j + k;

	  map_in[n] = BOUNDARY;
	  map_h[n] = 0.0;
	  /* Fluid if r(x,y) <= capillary width (L/2) */
	  r = sqrt(x*x + y*y);
	  if (r <= rc) {
	    map_in[n] = FLUID;
	  }

	  if (map_in[n] == BOUNDARY) {
	    nsolid++;
	    if (k >= z1 && k <= z2) {
	      map_h[n] = H;
	    }
	  }
	}
      }
    }

  }
  else {
    /* Square */

    for (i = 0; i < xmax; i++) {
      for (j = 0; j < ymax; j++) {
	for (k = 0; k < zmax; k++) {

	  n = ymax*zmax*i + zmax*j + k;

	  map_in[n] = FLUID;
	  
	  if (i == 0 || j == 0 || i == xmax - 1 || j == ymax - 1) {
	    map_in[n] = BOUNDARY;
	  }

	  if (map_in[n] == BOUNDARY) {
	    nsolid++;
	    if (k >= z1 && k <= z2) {
	      map_h[n] = H;
	    }
	  }
	}
      }
    }
  }

  /* picture */

  printf("\nCross section (%d = fluid, %d = solid)\n", FLUID, BOUNDARY);

  k = 0;
  for (i = 0; i < xmax; i++) {
    for (j = 0; j < ymax; j++) {
	n = ymax*zmax*i + zmax*j + k;
      
	if (map_in[n] == BOUNDARY) printf(" %d", BOUNDARY);
	if (map_in[n] == FLUID)    printf(" %d", FLUID);
    }
    printf("\n");
  }


  printf("n = %d nsolid = %d nfluid = %d\n", xmax*ymax*zmax, nsolid,
	 xmax*ymax*zmax - nsolid);

  /* Write new data as char */

  fp_orig = fopen(filename, "w");
  if (fp_orig == NULL) {
    printf("Cant open output\n");
    exit(-1);
  }

  for (i = 0; i < xmax; i++) {
    for (j = 0; j < ymax; j++) {
      for (k = 0; k < zmax; k++) {
	n = ymax*zmax*i + zmax*j + k;

	fputc(map_in[n], fp_orig);
	if (output_type == STATUS_WITH_H) {
	  fwrite(map_h + n, sizeof(double), 1, fp_orig);
	}
      }
    }
  }

  fclose(fp_orig);

  free(map_in);

  return 0;
}

/*****************************************************************************
 *
 *  profile
 *
 *  This attempts to give a profile of the interface as a function
 *  of the radial distance from the centre of the capillary defined
 *  above.
 *
 *  For each position (i, j) we examine the 1-d profile phi(z) and
 *  locate the position of zero by linear interpolation. The results
 *  go to standard output.
 *
 *  Note that the test for the interface assumes the phi = -1 block
 *  is in the middle of the system (see block initialisation in
 *  src/phi_stats.c).
 *
 *****************************************************************************/

static void profile(const char * filename) {

  int ic, jc, kc, index;
  int inside;
  double rc = 0.5*(xmax-2);
  double x0 = 0.5*xmax + 0.5;
  double y0 = 0.5*ymax + 0.5;
  double r, x, y;
  double * phi;
  FILE * fp;

  phi = (double *) malloc(xmax*ymax*zmax*sizeof(double));
  if (phi == NULL) {
    printf("malloc(phi) failed\n");
    exit(-1);
  }

  /* Read the data */

  printf("Reading phi data from %s...\n", filename);

  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Failed to open %s\n", filename);
    exit(-1);
  }

  for (ic = 0; ic < xmax; ic++) {
    for (jc = 0; jc < ymax; jc++) {
      for (kc = 0; kc < zmax; kc++) {
	index = ic*zmax*ymax + jc*zmax + kc;
	fread(phi + index, 1, sizeof(double), fp);
      }
    }
  }

  fclose(fp);

  /* Find the interface for each solid location */

  for (ic = 0; ic < xmax; ic++) {
    x = 1.0 + ic - x0;
    for (jc = 0; jc < ymax; jc++) {
      y = 1.0 + jc - y0;

      r = sqrt(x*x + y*y);

      /* Work out whether solid or fluid */
      inside = 0;
      if (xsection == SQUARE) {
	if (ic > 0 && ic < xmax-1 && jc > 0 && jc < ymax-1) inside = 1;
      }
      if (xsection == CIRCLE) {
	if (r <= rc) inside = 1;
      }

      if (inside) {
	/* Examine the profile */
	double h, dh;
	h = -1.0;

	for (kc = z1; kc <= z2; kc++) {
	  index = ic*zmax*ymax + jc*zmax + kc;
	  if (phi[index] > 0.0 && phi[index+1] < 0.0) {
	    /* Linear interpolation to get surface position */
	    dh = phi[index] / (phi[index] - phi[index+1]);
	    h = 1.0 + kc + dh;
	  }
	}
	printf("%f %f\n", r, h);
      }
    }
  }

  free(phi);

  /* Do not return! */
  exit(0);

  return;
}
