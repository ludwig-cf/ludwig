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
 *  For compilation instructions see the Makefile.
 *
 *  $ make extract_colloids
 *
 *  $ ./a.out <colloid file name stub> <nfile> <csv file name>
 *
 *  where the
 *    
 *  1st argument is the file name stub (in front of the last dot),
 *  2nd argument is the number of parallel files (as set with XXX_io_grid),
 *  3rd argyment is the (single) ouput file name.
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
 *  (c) 2012-2019 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "colloid.h"

#define NX 32
#define NY 32
#define NZ 32

static const int  iread_ascii = 1;  /* Read ascii or binary (default) */
static const int  include_ref = 0;  /* Include reference colloids at far x-,y-,z-corners */
static const int  id = 1;  	    /* Output colloid id */
static const int  cds_with_m  = 0;  /* Output coordinate and orientation */
static const int  cds_with_v  = 0;  /* Output coordinate, velocity vector and magnitude */
static const int  cds_with_both  = 1;  /* Output both */

static const char * format3_    = "%10.5f, %10.5f, %10.5f, ";
static const char * format3end_ = "%10.5f, %10.5f, %10.5f\n";
static const char * formate4end_ = "%14.6e, %14.6e, %14.6e, %14.6e\n";
static const char * format7end_ = "%10.5f, %10.5f, %10.5f, %14.6e, %14.6e, %14.6e, %14.6e\n";
static const char * format10end_ = "%10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %14.6e, %14.6e, %14.6e, %14.6e\n";

void colloids_to_csv_header(FILE * fp);
void colloids_to_csv_header_with_m(FILE * fp);
void colloids_to_csv_header_with_v(FILE * fp);
void colloids_to_csv_header_with_both(FILE * fp);

int main(int argc, char ** argv) {

  int n;
  int nf, nfile;
  int ncolloid;
  int nread;
  int ncount = 0;
  
  double normv;

  colloid_state_t s1;
  colloid_state_t s2;

  FILE * fp_colloids = NULL;
  FILE * fp_csv = NULL;
  char filename[FILENAME_MAX];

  if (argc < 3) {
    printf("Usage: %s <colloid_datafile_stub> <no_of_files> <csv_filename>\n",
	   argv[0]);
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

  if (cds_with_m) colloids_to_csv_header_with_m(fp_csv);
  if (cds_with_v) colloids_to_csv_header_with_v(fp_csv);
  if (cds_with_both) colloids_to_csv_header_with_both(fp_csv);

  for (nf = 1; nf <= nfile; nf++) {

    /* We expect extensions 00n-001 00n-002 ... 00n-00n */ 

    snprintf(filename, sizeof(filename), "%s.%3.3d-%3.3d", argv[1], nfile, nf);
    printf("Filename: %s\n", filename);

    fp_colloids = fopen(filename, "r");


    if (fp_colloids == NULL) {
        printf("fopen(%s) failed\n", filename);
        exit(0);
    }

    if (iread_ascii) {
      nread = fscanf(fp_colloids, "%d22\n", &ncolloid);
      assert(nread == 1);
    }
    else {
      nread = fread(&ncolloid, sizeof(int), 1, fp_colloids);
      assert(nread == 1);
    }
    if (nread != 1) printf("Warning: problem reading number of collloids\n");

    printf("Reading %d colloids from %s\n", ncolloid, argv[1]);

    /* Read and rewrite the data */

    for (n = 0; n < ncolloid; n++) {

      if (iread_ascii) {
	colloid_state_read_ascii(&s1, fp_colloids);
      }
      else {
	colloid_state_read_binary(&s1, fp_colloids);
      }

      /* Offset the positions */
      s2.r[0] = s1.r[0] - 1.0;
      s2.r[1] = s1.r[1] - 1.0;
      s2.r[2] = s1.r[2] - 1.0;

      /* Write coordinates and orientation 's' or velocity */
      if (id) fprintf(fp_csv, "%4d, ", s1.index);
      fprintf(fp_csv, format3_, s2.r[0], s2.r[1], s2.r[2]);
      if (cds_with_m) fprintf(fp_csv, format3end_, s1.m[0], s1.m[1], s1.m[2]);
      if (cds_with_v) {
	normv = sqrt(s1.v[0]*s1.v[0] + s1.v[1]*s1.v[1] + s1.v[2]*s1.v[2]);
	fprintf(fp_csv, formate4end_, s1.v[0], s1.v[1], s1.v[2], normv);
      }
      if (cds_with_both) {
	normv = sqrt(s1.v[0]*s1.v[0] + s1.v[1]*s1.v[1] + s1.v[2]*s1.v[2]);
	fprintf(fp_csv, format7end_, s1.m[0], s1.m[1], s1.m[2], s1.v[0], s1.v[1], s1.v[2], normv);
      }
      ncount += 1;

    }
  }

  /* Finish colloid coordinate output */
  fclose(fp_csv);
  if (include_ref) {
    printf("Wrote %d actual colloids + 3 reference colloids in header\n",
	   ncount);
  }
  else {
    printf("Wrote %d colloids\n", ncount);
  }

  /* Finish */
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

  if (id) fprintf(fp, "%s", "id, ");
  fprintf(fp, "%s", "x, y, z\n");

  if (include_ref) {

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

  }

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

  if (id) fprintf(fp, "%s", "id, ");
  fprintf(fp, "%s", "x, y, z, mx, my, mz\n");

  if (include_ref) {

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

  }

  return;
}

/*****************************************************************************
 *
 *  colloids_to_csv_header_with_v
 *
 *****************************************************************************/

void colloids_to_csv_header_with_v(FILE * fp) {

  double r[3];

  if (id) fprintf(fp, "%s", "id, ");
  fprintf(fp, "%s", "x, y, z, vx, vy, vz, normv\n");

  if (include_ref) {

    r[0] = 1.0*NX - 1.0;
    r[1] = 0.0;
    r[2] = 0.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

    r[0] = 0.0;
    r[1] = 1.0*NY - 1.0;
    r[2] = 0.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

    r[0] = 0.0;
    r[1] = 0.0;
    r[2] = 1.0*NZ - 1.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

  }

  return;
}


/*****************************************************************************
 *
 *  colloids_to_csv_header_with_both
 *
 *****************************************************************************/

void colloids_to_csv_header_with_both(FILE * fp) {

  double r[3];

  if (id) fprintf(fp, "%s", "id, ");
  fprintf(fp, "%s", "x, y, z, mx, my, mz, vx, vy, vz, normv\n");

  if (include_ref) {

    r[0] = 1.0*NX - 1.0;
    r[1] = 0.0;
    r[2] = 0.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

    r[0] = 0.0;
    r[1] = 1.0*NY - 1.0;
    r[2] = 0.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

    r[0] = 0.0;
    r[1] = 0.0;
    r[2] = 1.0*NZ - 1.0;

    fprintf(fp, format3_, r[0], r[1], r[2]);
    fprintf(fp, format3end_, 0, 0, 0, 0);

  }

  return;
}
