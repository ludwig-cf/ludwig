/*****************************************************************************
 *
 *  extract_colloids.c
 *
 *  Convert an output file to a csv file suitable for Paraview.
 *  The csv file uses three extra particles at (xmax, 0, 0)
 *  (0, ymax, 0) and (0, 0, zmax) to define the extent of the
 *  system.
 *
 *  If you want a different set of colloid properties, you need
 *  to arrange the header, and the output appropriately. The read
 *  can be ascii or binary and is set by the switch below.
 *
 *  Compile against the ludwig library, e.g.,
 *  $ cd ../src
 *  $ make lib
 *  $ cd ../util
 *  $ $(CC) -I../mpi_s -I../src extract_colloids.c \
 *          -L../mpi_s -lmpi -L../src -lludwig -lm
 *
 *  $ ./a.out <colloid file name> <nfile> <csv file name>
 *
 *  where the first argument is the file name stub
 *        the second is the number of parallel files
 *        and the third is the (single) ouput file name
 *
 *  If you have a set of files, try (eg. here with 4 parallel output files),
 *
 *  $ for f in config.cds*004-001; do g=`echo $f | sed s/.004-001//`; \
 *  echo $g; ~/ludwig/trunk/util/extract_colloids $g 4 $g.csv; done
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

#define NX 256
#define NY 256
#define NZ 256

static const int    iread_ascii = 0;      /* Read ascii or binary */

static const char * format3_    = "%10.5f, %10.5f, %10.5f,";
static const char * format3end_ = "%10.5f, %10.5f, %10.5f\n";

void colloids_to_csv_header(FILE * fp);
void colloids_to_csv_header_with_m(FILE * fp);

int main(int argc, char ** argv) {

  int n, np;
  int nf, nfile;
  int ncolloid;
  int ncount = 0;

  colloid_state_t s1;
  colloid_state_t s2;

  FILE * fp_colloids;
  FILE * fp_csv;
  char filename[FILENAME_MAX];

  if (argc != 4) {
    printf("Usage: %s <colloid file> <n file> <csv file>\n", argv[0]);
    exit(0);
  }

  nfile = atoi(argv[2]);
  printf("Number of files: %d\n", nfile);

  /* Open csv file */

  fp_csv = fopen(argv[3], "w");

  if (fp_csv == NULL) {
    printf("fopen(%s) failed\n", argv[2]);
    exit(0);
  }

  colloids_to_csv_header_with_m(fp_csv);

  for (nf = 1; nf <= nfile; nf++) {

    /* We expect extensions 00n-001 00n-002 ... 00n-00n */ 

    sprintf(filename, "%s.%3.3d-%3.3d", argv[1], nfile, nf);
    printf("Filename: %s\n", filename);

    fp_colloids = fopen(filename, "r");


    if (fp_colloids == NULL) {
        printf("fopen(%s) failed\n", filename);
        exit(0);
    }

    if (iread_ascii) {
      fscanf(fp_colloids, "%d22\n", &ncolloid);
    }
    else {
      fread(&ncolloid, sizeof(int), 1, fp_colloids);
    }

    printf("Reading %d colloids from %s\n", ncolloid, argv[1]);

    /* Read and rewrite the data */

    for (n = 0; n < ncolloid; n++) {

      if (iread_ascii) {
	colloid_state_read_ascii(&s1, fp_colloids);
      }
      else {
	colloid_state_read_binary(&s1, fp_colloids);
      }

      /* Reverse coordinates and offset the positions */

      s2.r[0] = s1.r[2] - 0.5;
      s2.r[1] = s1.r[1] - 0.5;
      s2.r[2] = s1.r[0] - 0.5;

      fprintf(fp_csv, format3_, s2.r[0], s2.r[1], s2.r[2]);
      fprintf(fp_csv, format3end_, s1.s[0], s1.s[1], s1.s[2]);
      ncount += 1;
    }
  }

  /* Finish. */

  fclose(fp_csv);
  fclose(fp_colloids);

  printf("Wrote %d actual colloids + 3 reference colloids in header\n", ncount);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_to_csv_header
 *
 *****************************************************************************/

void colloids_to_csv_header(FILE * fp) {

  double r[3];

  fprintf(fp, "%s", "x, y, z\n");

  r[0] = 1.0*NX - 1.0;
  r[1] = 0.0;
  r[2] = 0.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, "\n");

  r[0] = 0.0;
  r[1] = 1.0*NY - 1.0;
  r[2] = 0.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, "\n");

  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 1.0*NZ - 1.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, "\n");

  return;
}

/*****************************************************************************
 *
 *  colloids_to_csv_header_with_m
 *
 *****************************************************************************/

void colloids_to_csv_header_with_m(FILE * fp) {

  double r[3];
  double m[3];

  fprintf(fp, "%s", "x, y, z, mx, my, mz\n");

  r[0] = 1.0*NX - 1.0;
  r[1] = 0.0;
  r[2] = 0.0;

  m[0] = 1.0;
  m[1] = 0.0;
  m[2] = 0.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, format3end_, m[0], m[1], m[2]);

  r[0] = 0.0;
  r[1] = 1.0*NY - 1.0;
  r[2] = 0.0;

  m[0] = 0.0;
  m[1] = 1.0;
  m[2] = 0.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, format3end_, m[0], m[1], m[2]);

  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 1.0*NZ - 1.0;

  m[0] = 0.0;
  m[1] = 0.0;
  m[2] = 1.0;

  fprintf(fp, format3_, r[0], r[1], r[2]);
  fprintf(fp, format3end_, m[0], m[1], m[2]);

  return;
}
