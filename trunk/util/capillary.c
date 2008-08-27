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
 *  $Id: capillary.c,v 1.1.1.1 2008-08-27 13:48:17 kevin Exp $
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

const int xmax = 16;
const int ymax = 16;
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
const double H = 0.0;

/* WETTING. */
/* A section of capillary between z1 and z2 (inclusive) will have
 * wetting property H = H, the remainder H = 0 */

const int z1 = 1;
const int z2 = 16;

/* OUTPUT */
/* You can generate a file with solid/fluid status information only,
 * or one which includes the wetting parameter H. */

enum {STATUS_ONLY, STATUS_WITH_H};
const int output_type = STATUS_ONLY;

/* OUTPUT FILENAME */

const char * filename = "capillary.001-001";

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
