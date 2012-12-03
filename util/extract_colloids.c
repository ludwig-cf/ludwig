/*****************************************************************************
 *
 *  extract_colloids.c
 *
 *  Convert an output file to a csv file suitable for Paraview.
 *  The csv file uses three extra particles at (xmax, 0, 0)
 *  (0, ymax, 0) and (0, 0, zmax) to define the extend of the
 *  system.
 * 
 *  NOTE: This script is a first version. It does not work with 
 *        parallel input, i.e. more than one input file.
 *
 *  Compile against the ludwig library, e.g.,
 *  $ cd ../src
 *  $ make lib
 *  $ cd ../util
 *  $ $(CC) -I../mpi_s -I../src extract_colloids.c -L../mpi_s -lmpi -L../src -lludwig -lm
 *
 *  $ ./a.out <colloid file name> <csv file name>
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "colloid.h"

#define NX 128
#define NY 128
#define NZ 128

static const char * format_ = "%9.5f, %9.5f, %9.5f, %9.5f, %9.5f, %9.5f\n";

void colloids_to_csv_header(FILE * fp);

int main(int argc, char ** argv) {

  int n;
  int ncolloid;

  colloid_state_t s1;
  colloid_state_t s2;

  FILE * fp_colloids;
  FILE * fp_csv;

  if (argc != 3) {
    printf("Usage: %s <colloid file> <csv file>\n", argv[0]);
    exit(0);
  }

  /* Open existing file, and csv file */

  fp_colloids = fopen(argv[1], "r");
  fp_csv = fopen(argv[2], "w");

  if (fp_colloids == NULL) {
    printf("fopen(%s) failed\n", argv[1]);
    exit(0);
  }

  if (fp_csv == NULL) {
    printf("fopen(%s) failed\n", argv[2]);
    exit(0);
  }


  fscanf(fp_colloids, "%d22\n", &ncolloid);
  printf("Reading %d colloids from %s\n", ncolloid, argv[1]);

  colloids_to_csv_header(fp_csv);

  /* Read and rewrite the data */

  for (n = 0; n < ncolloid; n++) {
    colloid_state_read_ascii(&s1, fp_colloids);
    /* Reverse coordinates and offset the positions */
    s2.r[0] = s1.r[2] - 0.5;
    s2.r[1] = s1.r[1] - 0.5;
    s2.r[2] = s1.r[0] - 0.5;
    s2.m[0] = s1.m[2];
    s2.m[1] = s1.m[1];
    s2.m[2] = s1.m[0];
    fprintf(fp_csv, format_, s2.r[0], s2.r[1], s2.r[2],
	    s2.m[0], s2.m[1], s2.m[2]);
  }

  /* Finish. */

  fclose(fp_csv);
  fclose(fp_colloids);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_to_csv_header
 *
 *****************************************************************************/

void colloids_to_csv_header(FILE * fp) {

  double r[3];
  double m[3];

  fprintf(fp, "%s", "x, y, z, mx, my, mz\n");

  r[0] = 1.0*NX - 1.0;
  r[1] = 0.0;
  r[2] = 0.0;

  m[0] = 1.0;
  m[1] = 0.0;
  m[2] = 0.0;

  fprintf(fp, format_, r[0], r[1], r[2], m[0], m[1], m[2]);

  r[0] = 0.0;
  r[1] = 1.0*NY - 1.0;
  r[2] = 0.0;

  m[0] = 0.0;
  m[1] = 1.0;
  m[2] = 0.0;

  fprintf(fp, format_, r[0], r[1], r[2], m[0], m[1], m[2]);

  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 1.0*NZ - 1.0;

  m[0] = 0.0;
  m[1] = 0.0;
  m[2] = 1.0;

  fprintf(fp, format_, r[0], r[1], r[2], m[0], m[1], m[2]);

  return;
}
