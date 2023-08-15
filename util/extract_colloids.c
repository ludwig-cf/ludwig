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
 *  $ ./extract_colloids <colloid file name>
 *
 *  The file name must be of the form "config.cds00020000.002-001";
 *  if there are more than one file (from parallel i/o), any
 *  individual file will do, e.g., the first one.
 *
 *  The corresponding output will be a single file with combined information:
 *  colloids-00020000.csv for the example above.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "colloid.h"
#include "util_fopen.h"
#include "util.h"
#include "util_ellipsoid.h"
#include "util_vector.h"

#define NX 32
#define NY 32
#define NZ 32

static const int  iread_ascii = 1;  /* Read ascii or binary (default) */
static const int  include_ref = 0;  /* Include reference colloids at far x-,y-,z-corners */
static const int  id = 1;  	    /* Output colloid id */
static const int  cds_with_m  = 0;  /* Output coordinate and orientation */
static const int  cds_with_v  = 1;  /* Output coordinate, velocity vector and magnitude */

static const char * format3_    = "%10.5f, %10.5f, %10.5f, ";
static const char * format3end_ = "%10.5f, %10.5f, %10.5f\n";
static const char * format3wocomma_    = "%10.5f %10.5f %10.5f ";
static const char * format3wocommaend_ = "%10.5f %10.5f %10.5f\n";
static const char * formate4end_ = "%14.6e, %14.6e, %14.6e, %14.6e\n";

void colloids_to_csv_header(FILE * fp);
void colloids_to_csv_header_with_m(FILE * fp);
void colloids_to_csv_header_with_v(FILE * fp);
void colloids_to_vtk_header(FILE * fp);
void colloids_to_vtk_inbetween(FILE * fp);
int file_name_to_ntime(const char * filename);
int file_name_to_nfile(const char * filename);

int main(int argc, char ** argv) {

  int n;
  int nf, nfile;
  int ncolloid;
  int nread;
  int ntime = 0;
  int ncount = 0;
  double worldv1[3]={1.0,0.0,0.0};
  double worldv2[3]={0.0,1.0,0.0};
  double elev1[3], elev2[3], elev3[3];
  double *quater;

  double normv;

  colloid_state_t s1;
  colloid_state_t s2;

  FILE * fp_colloids = NULL;
  FILE * fp_csv = NULL;
  char csv_filename[BUFSIZ] = {0};
  FILE * fp_vtk = NULL;
  char vtk_filename[BUFSIZ] = {0};

  if (argc < 2) {
    printf("Usage: %s <colloid_datafile>\n", argv[0]);
    exit(0);
  }

  /* Check the file name. */
  ntime = file_name_to_ntime(argv[1]);
  nfile = file_name_to_nfile(argv[1]);
  printf("Time step:       %d\n", ntime);
  printf("Number of files: %d\n", nfile);

  /* Open csv file (output) */

  sprintf(csv_filename, "colloids-%8.8d.csv", ntime);
  fp_csv = util_fopen(csv_filename, "w");


  if (fp_csv == NULL) {
    printf("fopen(%s) failed\n", argv[3]);
    exit(0);
  }

  /* Open vtk file (output) */

  sprintf(vtk_filename, "colloids-%8.8d.vtk", ntime);
  fp_vtk = util_fopen(vtk_filename, "w");


  if (fp_csv == NULL) {
    printf("fopen(%s) failed\n", argv[3]);
    exit(0);
  }

  if (cds_with_m) colloids_to_csv_header_with_m(fp_csv);
  if (cds_with_v) colloids_to_csv_header_with_v(fp_csv);

  colloids_to_vtk_header(fp_vtk);

  for (nf = 1; nf <= nfile; nf++) {

    char filename[BUFSIZ] = {0};

    /* We expect extensions 00n-001 00n-002 ... 00n-00n */

    sprintf(filename, "config.cds%8.8d.%3.3d-%3.3d", ntime, nfile, nf);
    printf("Filename: %s\n", filename);

    fp_colloids = util_fopen(filename, "r");


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
      s2.r[0] = s1.r[0] - 0.5;
      s2.r[1] = s1.r[1] - 0.5;
      s2.r[2] = s1.r[2] - 0.5;

      /* Write coordinates and orientation 's' or velocity */
      if (id) fprintf(fp_csv, "%4d, ", s1.index);
      fprintf(fp_csv, format3_, s2.r[0], s2.r[1], s2.r[2]);
      if (cds_with_m) fprintf(fp_csv, format3end_, s1.s[0], s1.s[1], s1.s[2]);
      if (cds_with_v) {
	normv = sqrt(s1.v[0]*s1.v[0] + s1.v[1]*s1.v[1] + s1.v[2]*s1.v[2]);
	fprintf(fp_csv, formate4end_, s1.v[0], s1.v[1], s1.v[2], normv);
      }
      /* Write coordinates to vtk file*/
      fprintf(fp_vtk, format3wocommaend_, s1.r[0]-1.0, s1.r[1]-1.0, s1.r[2]-1.0);
      /* Write Orientations to vtk file*/
      colloids_to_vtk_inbetween(fp_vtk);
      quater = s1.quater;
      util_q4_rotate_vector(quater, worldv1, elev1);
      util_q4_rotate_vector(quater, worldv2, elev2);
      cross_product(elev1,elev2,elev3);
      util_vector_normalise(3, elev3);
      fprintf(fp_vtk, format3wocomma_, 2.0*s1.elabc[0]*elev1[0], 2.0*s1.elabc[0]*elev1[1],2.0*s1.elabc[0]*elev1[2]);
      fprintf(fp_vtk, format3wocomma_, 2.0*s1.elabc[1]*elev2[0], 2.0*s1.elabc[1]*elev2[1], 2.0*s1.elabc[1]*elev2[2]);
      fprintf(fp_vtk, format3wocommaend_, 2.0*s1.elabc[2]*elev3[0], 2.0*s1.elabc[2]*elev3[1], 2.0*s1.elabc[2]*elev3[2]);
      ncount += 1;

    }

  }

  /* Finish colloid coordinate output */
  fclose(fp_csv);
  fclose(fp_vtk);
  if (include_ref) {
    printf("Wrote %d actual colloids + 3 reference colloids in header to %s\n",
	   ncount, csv_filename);
  }
  else {
    printf("Wrote %d colloids to %s\n", ncount, csv_filename);
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
 *  colloids_to_vtk_header
 *
 *****************************************************************************/

void colloids_to_vtk_header(FILE * fp) {

  assert(fp);

  fprintf(fp, "# vtk DataFile Version 5.1\n");
  fprintf(fp, "vtk output\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET POLYDATA\n");
  fprintf(fp, "POINTS 2 float\n");
  fprintf(fp, "%d %d %d\n", 0, 0, 0);

 }

/*****************************************************************************
 *
 *  colloids_to_vtk_inbetween
 *
 *****************************************************************************/

void colloids_to_vtk_inbetween(FILE * fp) {

  assert(fp);

  fprintf(fp, "POINT_DATA %d\n", 2);
  fprintf(fp, "TENSORS tensors double\n");
  fprintf(fp, "%d %d %d ", 0, 0, 0);
  fprintf(fp, "%d %d %d ", 0, 0, 0);
  fprintf(fp, "%d %d %d \n", 0, 0, 0);

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
 *  file_name_to_nfile
 *
 *  The file name must be of the form "config.cds00020000.004-001".
 *
 *****************************************************************************/

int file_name_to_nfile(const char * filename) {

  int nfile = 0;
  const char * ext = strrchr(filename, '.'); /* last dot */

  if (ext != NULL) {
    char buf[BUFSIZ] = {0};
    strncpy(buf, ext + 1, 3);
    nfile = atoi(buf);
  }

  return nfile;
}

/*****************************************************************************
 *
 * file_name_to_ntime
 *
 *  The file name must be of the form "config.cds00020000.004-001".
 *
 *****************************************************************************/

int file_name_to_ntime(const char * filename) {

  int ntime = -1;
  const char * tmp = strchr(filename, 's'); /* Must be "config.cds" */

  if (tmp) {
    char buf[BUFSIZ] = {0};
    strncpy(buf, tmp + 1, 8);
    ntime = atoi(buf);
  }

  if (0 > ntime || ntime >= 1000*1000*1000) {
    printf("Could not parse a time step from file name %s\n", filename);
    exit(-1);
  }

  return ntime;
}
