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
 *  3rd argument is the (single) ouput file name.
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
 *  (c) 2012-2022 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "colloid.h"
#include "util_fopen.h"

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
static const char * formate4end_ = "%14.6e, %14.6e, %14.6e, %14.6e\n";

void colloids_to_csv_header(FILE * fp);
void colloids_to_csv_header_with_m(FILE * fp);
void colloids_to_csv_header_with_v(FILE * fp);
int util_io_posix_filename(const char * input, char * buf, size_t bufsz);

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

  if (argc < 4) {
    printf("Usage: %s <colloid_datafile_stub> <no_of_files> <csv_filename>\n",
	   argv[0]);
    exit(0);
  }

  nfile = atoi(argv[2]);
  printf("Number of files: %d\n", nfile);

  /* Open csv file (check input first) */
  {
    char csv_filename[BUFSIZ] = {0};
    int ireturn = util_io_posix_filename(argv[3], csv_filename, BUFSIZ);
    if (ireturn == 0) {
      fp_csv = util_fopen(csv_filename, "w");
    }
    else {
      printf("Please use output filename with allowed characters\n");
      printf("[A-Z a-z 0-9 - _ .] only\n");
      exit(-1);
    }
  }

  if (fp_csv == NULL) {
    printf("fopen(%s) failed\n", argv[3]);
    exit(0);
  }

  if (cds_with_m) colloids_to_csv_header_with_m(fp_csv);
  if (cds_with_v) colloids_to_csv_header_with_v(fp_csv);

  for (nf = 1; nf <= nfile; nf++) {

    /* We expect extensions 00n-001 00n-002 ... 00n-00n */ 

    snprintf(filename, sizeof(filename), "%s.%3.3d-%3.3d", argv[1], nfile, nf);
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
 *  util_io_posix_filename
 *
 *  A posix "fully portable filename" (not a path) has allowed characters:
 *   A-Z, a-z, 0-9, ".", "-", and "_"
 *  The first character must not be a hyphen.
 *
 *  This returns a positive value if a replacement is required,
 *  a negative value if the supplied buffer is too small,
 *  or zero on a succcessful copy with no replacements.
 *
 *  The output buffer contains the input with any duff characters replaced
 *  by "_".
 *
 *****************************************************************************/

int util_io_posix_filename(const char * input, char * buf, size_t bufsz) {

  int ifail = 0;
  size_t len = strnlen(input, FILENAME_MAX-1);
  const char replacement_character = '_';

  if (bufsz <= len) {
    /* would be truncated */
    ifail = -1;
  }
  else {
    /* Copy, but replace anything that's not posix */
    for (size_t i = 0; i < len; i++) {
      char c = input[i];
      if (i == 0 && c == '-') {
	/* Replace */
	buf[i] = replacement_character;
	ifail += 1;
      }
      else if (isalnum(c) || c == '_' || c == '-' || c == '.') {
	/* ok */
	buf[i] = input[i];
      }
      else {
	/* Replace */
	buf[i] = replacement_character;
	ifail += 1;
      }
    }
    /* Terminate */
    buf[len] = '\0';
  }

  return ifail;
}
