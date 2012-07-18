/*****************************************************************************
 *
 *  process.c
 *
 *  For a porous media data set (e.g., X-ray tomography) stored as
 *  ASCII (0 = fluid, 1 = solid), we generate a file suitable for
 *  input to Ludwig.
 *
 *  There are two main concerns:
 *    1) the structure must be periodic in the flow direction, meaning
 *       a 'reflection' is required. The flow direction is assumed to
 *       be the x-direction.
 *
 *    2) Similarly, the y-direction and z-direction must be periodic.
 *       To satify this, we pad the edges with solid sites.
 *
 *  The input file is assumed to have one data value per line, with
 *  the z-direction running fastest.
 *
 *  The output is a binary suitable for input to Ludwig.
 *
 *   Usage:
 *      ./a.out input_file output_file
 *
 *  $Id: process.c,v 1.3 2008-10-22 16:16:23 kevin Exp $
 *  
 *  Edinburgh Soft Matter and Statistcal Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/site_map.h"

/* SYSTEM SIZE */
/* This is the size of the original data to be read. */

const int xinput = 64;
const int yinput = 64;
const int zinput = 64;

/* REFLECTION */
/* Set this flag is a reflection of the data in the x-direction is
 * required. */

enum {REFLECTION, NO_REFLECTION};
const int reflect = NO_REFLECTION;

/* PADDING */
/* PAD_NONE keeps the original boundaries.
 * PAD_OVERWRITE overwrites whatever is at the edge with solid value,
 * so it keeps the same system size.
 * PAD_ADD appends one solid site at each edge, so the system is 2 sites
 * wider in the y-direction and z-direction on output. */

enum {PAD_NONE, PAD_OVERWRITE, PAD_ADD};
const int pad = PAD_NONE;

int main (int argc, char ** argv) {

  char * data_in;
  char * data_out;
  int xout, yout, zout;
  int ic, jc, kc, index1, index2;
  int ntotal, nfluid, nsolid;
  int n, tmp;
  FILE * fp_file;

  if (argc != 3) {
    printf("Usage: %s input_file output_file\n", argv[0]);
    exit(-1);
  }

  /* Read the original file */

  fp_file = fopen(argv[1], "r");
  if (fp_file == NULL) {
    printf("Failed to open %s\n", argv[1]);
    exit(-1);
  }

  data_in = (char *) malloc(xinput*yinput*zinput*sizeof(char));
  if (data_in == NULL) {
    printf("malloc(data_in) failed\n");
    exit(-1);
  }

  printf("Reading the original file: %s\n", argv[1]);
  printf("Size x-direction: %4d\n", xinput);
  printf("Size y-direction: %4d\n", yinput);
  printf("Size z-direction: %4d\n", zinput);

  ntotal = 0;
  nfluid = 0;
  nsolid = 0;

  for (ic = 0; ic < xinput; ic++) {
    for (jc = 0; jc < yinput; jc++) {
      for (kc = 0; kc < zinput; kc++) {
	index1 = yinput*zinput*ic + zinput*jc + kc;
	n = fscanf(fp_file, "%d\n", &tmp);
	if (n != 1) printf("Error reading position %d %d %d\n", ic, jc, kc);
	ntotal++;
	if (tmp == 0) {
	  data_in[index1] = FLUID;
	  nfluid++;
	}
	else {
	  data_in[index1] = BOUNDARY;
	  nsolid++;
	}
      }
    }
  }

  fclose(fp_file);
  printf("Finished reading input file...\n");
  printf("Total sites: %8d\n", ntotal);
  printf("Fluid sites: %8d\n", nfluid);
  printf("Solid sites: %8d\n", nsolid);
  if (ntotal != nfluid + nsolid) printf("** Inconsistent site count!\n");
  printf("\n");

  /* Set the output size */

  printf("Output system...\n");

  if (reflect == NO_REFLECTION) {
    xout = xinput;
    printf("The system will not be reflected.\n");
  }
  if (reflect == REFLECTION) {
    xout = 2*xinput;
    printf("The system will be reflected.\n");
  }

  if (pad == PAD_NONE) {
    yout = yinput;
    zout = zinput;
    printf("No padding\n");
  }
  if (pad == PAD_OVERWRITE) {
    yout = yinput;
    zout = zinput;
    printf("Padding will overwrite\n");
  }
  if (pad == PAD_ADD) {
    yout = yinput + 2;
    zout = zinput + 2;
    printf("Padding will add\n");
  }

  data_out = (char *) malloc(xout*yout*zout*sizeof(char));
  if (data_out == NULL) {
    printf("malloc(data_out) has failed\n");
    exit(-1);
  }

  printf("Output size x-direction: %4d\n", xout);
  printf("Output size y-direction: %4d\n", yout);
  printf("Output size z-direction: %4d\n", zout);

  /* Copy the input to output with appropriate padding */

  for (ic = 0; ic < xinput; ic++) {
    for (jc = 0; jc < yout; jc++) {
      for (kc = 0; kc < zout; kc++) {
	index1 = yout*zout*ic + zout*jc + kc;
	data_out[index1] = BOUNDARY;
      }
    }
  }

  if (pad == PAD_NONE) {
    for (ic = 0; ic < xinput; ic++) {
      for (jc = 0; jc < yout; jc++) {
	for (kc = 0; kc < zout; kc++) {
	  index1 = yout*zout*ic + zout*jc + kc;
	  data_out[index1] = data_in[index1];
	}
      }
    }
  }

  if (pad == PAD_OVERWRITE) {
    for (ic = 0; ic < xinput; ic++) {
      for (jc = 1; jc < yout - 1; jc++) {
	for (kc = 1; kc < zout - 1; kc++) {
	  index1 = yout*zout*ic + zout*jc + kc;
	  data_out[index1] = data_in[index1];
	}
      }
    }
  }

  if (pad == PAD_ADD) {
    for (ic = 0; ic < xinput; ic++) {
      for (jc = 0; jc < yinput; jc++) {
	for (kc = 0; kc < zinput; kc++) {
	  index1 = yout*zout*ic + zout*(jc+1) + kc+1;
	  index2 = yinput*zinput*ic + zinput*jc + kc;
	  data_out[index1] = data_in[index2];
	}
      }
    }
  }

  /* Do the reflection if required. */

  for (ic = xinput; ic < xout; ic++) {
    for (jc = 0; jc < yout; jc++) {
      for (kc = 0; kc < zout; kc++) {
	index1 = yout*zout*ic + zout*jc + kc;
	index2 = yout*zout*(xout -  ic - 1) + zout*jc + kc;
	data_out[index1] = data_out[index2];
      }
    }
  }

  /* Write the final file */

  fp_file = fopen(argv[2], "w");
  if (fp_file == NULL) {
    printf("Failed to open %s\n", argv[2]);
  }

  ntotal = 0;
  nfluid = 0;
  nsolid = 0;

  for (ic = 0; ic < xout; ic++) {
    for (jc = 0; jc < yout; jc++) {
      for (kc = 0; kc < zout; kc++) {
	index1 = yout*zout*ic + zout*jc + kc;
	fputc(data_out[index1], fp_file);
	ntotal++;
	if (data_out[index1] == FLUID) nfluid++;
	if (data_out[index1] == BOUNDARY) nsolid++;
      }
    }
  }

  fclose(fp_file);
  printf("Total sites: %8d\n", ntotal);
  printf("Fluid sites: %8d\n", nfluid);
  printf("Solid sites: %8d\n", nsolid);
  if (ntotal != nfluid + nsolid) printf("** Inconsistent site count!\n");
  printf("\n");

  free(data_in);
  free(data_out);
  return 0;
}
