/****************************************************************************
 *
 *  extract.c
 *
 *  This is under revision as part of the move to new metadata.
 *  For most users, there will be no need to consolidate
 *  parallel output. Only post-processing is relevant (unrolling
 *  Lees Edwards plane, diagonalisation of liquid crystal order parameter).
 *
 *  ./extract data-file
 *
 *  The relevant metadata file is located from the file stub.
 *  Most options are still available (see below).
 *
 *  Older version ...
 *
 *  This program deals with the consolidation of parallel I/O
 *  (and in serial with Lees-Edwards planes). This program is
 *  serial.
 *
 *  To process/recombined parallel I/O into a single file with the
 *  correct order (ie., z running fastest, then y, then x) run
 *
 *  ./extract meta-file data-file
 *
 *  where 'meta-file' is the first meta information file in the series,
 *  and data file is the first datsa file for a given time step. All
 *  values are expected to be of data-type double - this will be
 *  preserved on output.
 *
 *  Clearly, the two input files must match, or results will be
 *  unpredictable.
 *
 *  COMMAND LINE OPTIONS
 *
 *  -a   Request ASCII output
 *  -b   Request binary output (the default)
 *  -i   Request coordinate indices in output (none by default)
 *  -k   Request VTK header STRUCTURED_POINTS (none by default)
 *  -l   Request VTK header RECTILINEAR_GRID
 *
 *  Options relevant to liquid crystal order parameter (only):
 *  -d   Request director output
 *  -s   Request scalar order parameter output
 *  -x   Request biaxial order parameter output
 *
 *  TO BUILD:
 *  Compile the serial version of Ludwig in  the usual way.
 *  In this directory:
 *
 *  $ make extract
 *
 *  should produce 'extract'
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *  University of Strathclyde, Glasgow, UK
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich  (oliver.henrich@strath.ac.uk)
 *
 *  (c) 2011-2023 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io_metadata.h"
#include "util.h"
#include "util_fopen.h"

#define MAXNTOTAL 1024 /* Maximum linear system size */

const int default_version = 2; /* Meta data version */
                               /* 1 = older output files */
                               /* 2 = current processor independent per file */

typedef enum vtk_enum {VTK_SCALARS, VTK_VECTORS} vtk_enum_t;
typedef struct metadata_v1_s metadata_v1_t;

struct metadata_v1_s {
  const char * stub;           /* file stub name */
  int nrbyte;                  /* byte per record */
  int is_bigendean;            /* flag */
  int npe;                     /* comm sz */
  int pe[3];                   /* mpi cart size */
  int coords[3];               /* mpi cart coords */
  int ntotal[3];               /* ntotal */
  int nlocal[3];               /* nlocal */
  int nplanes;                 /* number of le planes */
  int offset[3];               /* noffset CHECK */
  int nio;                     /* number of io groups (files) */
};

int ntargets[3];               /* Output target */
int io_size[3];                /* I/O decomposition */
int nrec_ = 1;
int input_isbigendian_ = -1;   /* May need to deal with endianness */
int reverse_byte_order_ = 0;   /* Switch for bigendian input */
int input_binary_ = -1;        /* Format of input detected from meta data */
int output_binary_ = 0;        /* Switch for format of final output */
int is_velocity_ = 0;          /* Switch to identify velocity field */
int output_index_ = 0;         /* For ASCII output, include (i,j,k) indices */
int output_vtk_ = 0;           /* Write ASCII VTK header */
                               /* 1 = STRUCTURED_POINTS 2 = RECTILINEAR_GRID */
int output_lcs_ = 0;
int output_lcd_ = 0;
int output_lcx_ = 0;

int le_t0_ = 0;                /* LE offset start time (time steps) */ 

int output_cmf_ = 0;           /* flag for output in column-major format */
int output_q_raw_ = 0;         /* 0 -> LC s, director, b otherwise raw q5 */

double le_speed_ = 0.0;
double le_displace_ = 0.0;
double * le_displacements_;
double * le_duy_;

int extract_driver(const char * filename, metadata_v1_t * meta, int version);
int read_version1(int ntime, metadata_v1_t * m, double * datasection);
int read_version2(int ntime, metadata_v1_t * m, double * datasection);

void read_meta_data_file(const char *, metadata_v1_t * metadata);
int  read_data_file_name(const char *);

int write_data(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_cmf(FILE * fp, int n[3], int nr0, int nr, double * data);
int write_data_ascii(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_ascii_cmf(FILE * fp, int n[3], int nrec0, int nrec, double *);
int write_data_binary(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_binary_cmf(FILE * fp, int n[3], int nrec0, int nrec, double *);

int site_index(int, int, int, const int nlocal[3]);
void read_data(FILE *, metadata_v1_t * meta, double *);

int write_vtk_header(FILE * fp, int nrec, int ndim[3], const char * descript,
		     vtk_enum_t vtk);
int write_vtk_header_points(FILE * fp, int nrec, int ndim[3],
			    const char * descript,
			    vtk_enum_t vtk);
int write_vtk_header_rectilinear(FILE * fp, int nrec, int ndim[3],
				 const char * descript,
				 vtk_enum_t vtk);

int write_qab_vtk(int ntime, int ntargets[3], double * datasection);

int copy_data(metadata_v1_t * metadata, double *, double *);
int le_displacement(int ntotal[3], int, int);
void le_set_displacements(metadata_v1_t * meta);
void le_unroll(metadata_v1_t * meta, double *);

int lc_transform_q5(int * nlocal, double * datasection);
int lc_compute_scalar_ops(double q[3][3], double qs[5]);

const char * file_stub_valid(const char * input);
int file_get_file_nfile(const char * filename);
int file_get_file_index(const char * filename);
int extract_process_and_output(const char * stub, int ntime,
			       io_metadata_t * meta);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int optind;
  int version = default_version;
  metadata_v1_t metadata = {0};

  MPI_Init(&argc, &argv);

  /* Check the command line, then parse the meta data information,
   * and sort out the data file name  */

  for (optind = 1; optind < argc && argv[optind][0] == '-'; optind++) {
    switch (argv[optind][1]) {
    case 'a':
      output_binary_ = 0; /* Request ASCII */ 
      break;
    case 'b':
      output_binary_ = 1; /* Request Binary */
      break;
    case 'd':
      output_lcd_ = 1;    /* Request liquid crystal director output */
      break;
    case 'i':
      output_index_ = 1;  /* Request ic, jc, kc indices */
      break;
    case 'k':
      output_vtk_ = 1;    /* Request VTK header */
      output_cmf_ = 1;    /* Request column-major format for Paraview */ 
      break;
    case 'l':
      output_vtk_ = 2;
      output_cmf_ = 1;    /* Request column-major format for Paraview */
      break;
    case 's':
      output_lcs_ = 1; /* Request liquid crystal scalar order parameter */
      break;
    case 'x':
      output_lcx_ = 1; /* Request liquid crystal biaxial order paramter */
      break;
    default:
      fprintf(stderr, "Unrecognised option: %s\n", argv[optind]);
      fprintf(stderr, "Usage: %s [-abk] meta-file data-file\n", argv[0]);
      exit(EXIT_FAILURE);
    }   
  }

  /* New metadata format: completely short circuit the original
     processing via a new driver */
  if (optind == argc - 1) {
    /* Must have a recognised metadata quantity */
    const char * stub = file_stub_valid(argv[argc-1]);
    if (stub == NULL) {
      printf("Unrecognised data quantity %s\n", argv[argc-1]);
      exit(-1);
    }
    else {
      pe_t * pe = NULL;
      char buf[FILENAME_MAX] = {0};
      char * filename = buf;
      int ifail = 0;
      int itime = 0;

      int ifile = file_get_file_index(argv[argc-1]);
      int nfile = file_get_file_nfile(argv[argc-1]);
      io_metadata_t * meta = NULL;

      printf("%s %s\n", argv[0], argv[argc-1]);
      printf("Identified file name as %s\n", stub);
      sprintf(filename, "%s-metadata.%3.3d-%3.3d", stub, nfile, ifile);
      printf("Attempt to read metadata file %s\n", filename);
      pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
      ifail = io_metadata_from_file(pe, filename, &meta);
      if (ifail != 0) printf("Failed to read/parse metadata file\n");

      itime = read_data_file_name(argv[argc-1]);
      ifail = extract_process_and_output(stub, itime, meta);
      /* Release meta resources */
      pe_free(pe);
      printf("Complete processing for %s\n", argv[argc-1]);
    }
    exit(0);
  }

  /* Continue with older invocation <metadata file> <data file> */
  if (optind > argc-2) {
    printf("Usage: %s [-abk] meta-file data-file\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  {
    /* Must have a recognised metadata quantity */
    const char * stub = file_stub_valid(argv[optind]);
    if (stub == NULL) {
      printf("Unrecognised metadata file %s\n", argv[optind]);
      exit(-1);
    }
    else {
      char buf[FILENAME_MAX] = {0};
      char * filename = buf;
      int ifile = file_get_file_index(argv[optind]);
      int nfile = file_get_file_nfile(argv[optind]);
      sprintf(filename, "%s.%3.3d-%3.3d.meta", stub, nfile, ifile);
      read_meta_data_file(filename, &metadata);
    }
  }

  extract_driver(argv[optind+1], &metadata, version);

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  extract_driver
 *
 *****************************************************************************/

int extract_driver(const char * filename, metadata_v1_t * meta, int version) {

  int ntime;
  int i, n;

  double * datasection;
  char io_data[FILENAME_MAX];
  const char * suf = ".vtk";

  FILE * fp_data;

  ntime = read_data_file_name(filename);
  assert(0 <= ntime && ntime < 1000*1000*1000);

  /* Work out parallel local file size */

  assert(io_size[0] == 1); /* No x-decompositions WILL FAIL */

  switch (version) {
  case 1:
    for (i = 0; i < 3; i++) {
     meta->nlocal[i] = meta->ntotal[i]/meta->pe[i];
    }
    break;
  case 2:
    for (i = 0; i < 3; i++) {
      meta->nlocal[i] = meta->ntotal[i]/io_size[i];
    }
    break;
  default:
    printf("Invalid version %d\n", version);
  }
    
  /* No. sites in the target section (always original at moment) */

  for (i = 0; i < 3; i++) {
    ntargets[i] = meta->ntotal[i];
  }
 
  n = nrec_*ntargets[0]*ntargets[1]*ntargets[2];
  datasection = (double *) calloc(n, sizeof(double));
  if (datasection == NULL) printf("calloc(datasection) failed\n");

  /* LE displacements as function of x */
  le_displace_ = le_speed_*(double) (ntime - le_t0_);
  n = meta->ntotal[0];
  if (n < 0 || n > MAXNTOTAL) {
    printf("Please check system size %d\n", meta->ntotal[0]);
    return -1;
  }
  le_displacements_ = (double *) calloc(n, sizeof(double));
  le_duy_ = (double *) calloc(n, sizeof(double));
  if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
  if (le_duy_ == NULL) printf("malloc(le_duy_) failed\n");
  le_set_displacements(meta);

  /* Read data file or files (version 2 uses v1 meta data for time being) */

  if (version == 1) read_version1(ntime, meta, datasection);
  if (version == 2) read_version2(ntime, meta, datasection);

  /* Unroll the data if Lees Edwards planes are present */

  if (meta->nplanes > 0) {
    printf("Unrolling LE planes from centre (displacement %f)\n",
	   le_displace_);
    le_unroll(meta, datasection);
  }

  if (nrec_ == 5 && strncmp(meta->stub, "q", 1) == 0) {

    if (output_vtk_ == 1 || output_vtk_ == 2) {
      /* We mandate this is the transformed Q_ab */
      lc_transform_q5(ntargets, datasection);
      write_qab_vtk(ntime, ntargets, datasection);
    }
    else {
      sprintf(io_data, "q-%8.8d", ntime);
      fp_data = util_fopen(io_data, "w+b");
      if (fp_data == NULL) {
	printf("fopen(%s) failed\n", io_data);
	exit(-1);
      }

      /* Here, we could respect the -d, -s, -x options. */
      /* But at the moment, always 5 components ... */
      if (output_q_raw_) {
	printf("Writing raw q to %s\n", io_data);
      }
      else {
	printf("Writing computed scalar q etc: %s\n", io_data);
	lc_transform_q5(ntargets, datasection);
      }

      if (output_cmf_ == 0) write_data(fp_data, ntargets, 0, 5, datasection);
      if (output_cmf_ == 1) write_data_cmf(fp_data, ntargets, 0, 5, datasection);

      fclose(fp_data);
    }
  }
  else if (nrec_ == 3 && strncmp(meta->stub, "fed", 3) == 0) {
    /* Require a more robust method to identify free energy densities */
    assert(0); /* Requires an update */
  }
  else {
    /* A direct input / output */

    /* Write a single file with the final section */

    snprintf(io_data, sizeof(io_data), "%s-%8.8d", meta->stub, ntime);

    if (output_vtk_ == 1 || output_vtk_ == 2) {

      strcat(io_data, suf);
      fp_data = util_fopen(io_data, "w+b");
      if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);
      printf("\nWriting result to %s\n", io_data);

      if (nrec_ == 3 && strncmp(meta->stub, "vel", 3) == 0) {
	write_vtk_header(fp_data, nrec_, ntargets, "velocity_field",
			 VTK_VECTORS);
      }
      else if (nrec_ == 1 && strncmp(meta->stub, "phi", 3) == 0) {
	write_vtk_header(fp_data, nrec_, ntargets, "composition",
			 VTK_SCALARS);
      }
      else {
	/* Assume scalars */
	write_vtk_header(fp_data, nrec_, ntargets, meta->stub, VTK_SCALARS);
      }

    }
    else {

      fp_data = util_fopen(io_data, "w+b");
      if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);
      printf("\nWriting result to %s\n", io_data);

    }

    if (output_cmf_ == 0) write_data(fp_data, ntargets, 0, nrec_, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntargets, 0, nrec_, datasection);

    fclose(fp_data);
  }

  free(le_displacements_);
  free(datasection);

  return 0;
}

/*****************************************************************************
 *
 *  Newer output files where each file has processor-independent order.
 *  PENDING update in meta data file output for more than 1 file.
 *
 *****************************************************************************/

int read_version2(int ntime, metadata_v1_t * meta, double * datasection) {

  int n;
  char io_data[BUFSIZ];

  double * datalocal = NULL;
  FILE * fp_data = NULL;

  assert(meta);
  assert(datasection);

  n = nrec_*meta->nlocal[0]*meta->nlocal[1]*meta->nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) {
    printf("calloc(datalocal) failed\n");
    exit(-1);
  }

  /* Main loop */

  for (n = 1; n <= meta->nio; n++) {

    /* Open the current data file */

    snprintf(io_data, BUFSIZ, "%s-%8.8d.%3.3d-%3.3d", meta->stub, ntime,
	     meta->nio, n);
    printf("-> %s\n", io_data);

    fp_data = util_fopen(io_data, "r+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    /* Read data file based on offsets recorded in the metadata,
     * then copy this section to the global array */

    read_data(fp_data, meta, datalocal);
    copy_data(meta, datalocal, datasection);

    fclose(fp_data);
  }

  free(datalocal);

  return 0;
}

/*****************************************************************************
 *
 *  Older output versions where each file has processor dependent
 *  order specified in the metadata file.
 *
 *****************************************************************************/

int read_version1(int ntime, metadata_v1_t * meta, double * datasection) {

  int ifail;
  int n;
  int p, pe_per_io;
  int rank;
  char io_metadata[BUFSIZ];
  char io_data[BUFSIZ];
  char line[BUFSIZ];
  char * pstr;

  double * datalocal = NULL;
  FILE * fp_metadata = NULL;
  FILE * fp_data = NULL;

  pe_per_io = meta->pe[0]*meta->pe[1]*meta->pe[2]/meta->nio;

  /* Allocate storage */

  n = nrec_*meta->nlocal[0]*meta->nlocal[1]*meta->nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) {
    printf("calloc(datalocal) failed\n");
    exit(-1);
  }

  /* Main loop */

  for (n = 1; n <= meta->nio; n++) {

    /* Open metadata file and skip 12 lines to get to the
     * decomposition inofmration */

    snprintf(io_metadata, sizeof(io_metadata), "%s.%3.3d-%3.3d.meta",
	     meta->stub, meta->nio, n);
    printf("Reading metadata file ... %s ", io_metadata);

    fp_metadata = util_fopen(io_metadata, "r");
    if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

    for (p = 0; p < 12; p++) {
      pstr = fgets(line, FILENAME_MAX, fp_metadata);
      if (pstr == NULL) printf("Failed to read line\n");
      printf("%s", line);
    }

    /* Open the current data file */

    snprintf(io_data, sizeof(io_data), "%s-%8.8d.%3.3d-%3.3d", meta->stub,
	     ntime, meta->nio, n);
    printf("-> %s\n", io_data);

    fp_data = util_fopen(io_data, "r+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    /* Read data file based on offsets recorded in the metadata,
     * then copy this section to the global array */

    for (p = 0; p < pe_per_io; p++) {

      ifail = fscanf(fp_metadata, "%d %d %d %d %d %d %d %d %d %d",
		     &rank,
		     &meta->coords[0], &meta->coords[1], &meta->coords[2],
		     &meta->ntotal[0], &meta->ntotal[1], &meta->ntotal[2],
		     &meta->offset[0], &meta->offset[1], &meta->offset[2]);
      if (ifail != 10) {
	printf("Meta data rank ... not corrrect\n");
	exit(-1);
      }

      read_data(fp_data, meta, datalocal);
      copy_data(meta, datalocal, datasection);
    }

    fclose(fp_data);
    fclose(fp_metadata);
  }

  free(datalocal);

  return 0;
}

/****************************************************************************
 *
 *  read_meta_data_file
 *
 *  This sets a number of the global variable from the meta data file.
 *
 ****************************************************************************/

void read_meta_data_file(const char * filename, metadata_v1_t * meta) {

  int nrbyte;
  int ifail;
  char tmp[FILENAME_MAX];
  char buf[BUFSIZ] = {0};
  char * p;
  FILE * fp_meta;
  const int ncharoffset = 33;

  assert(filename);
  assert(meta);

  fp_meta = util_fopen(filename, "r");
  if (fp_meta == NULL) {
    printf("fopen(%s) failed\n", filename);
    exit(-1);
  }

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%s\n", buf);
  if (ifail != 1) {
    printf("Meta data stub not read correctly\n");
    exit(-1);
  }
  meta->stub = file_stub_valid(buf);
  if (meta->stub == NULL) {
    printf("Meta data stub not recognised: %s\n", buf);
  }
  printf("Read stub: %s\n", meta->stub);

  /* Data description */
  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &nrbyte);
  assert(ifail == 1);
  printf("Record size (bytes): %d\n", nrbyte);

  /* PENDING: this deals with different formats until improved meta data
   * is available */

  if ((nrbyte % 8) == 0) {
    /* We have a binary file */
    input_binary_ = 1;
    nrec_ = nrbyte / 8;
  }
  else {
    /* ASCII: approx 22 characters per record */
    input_binary_ = 0;
    nrec_ = nrbyte / 22;
    assert(nrec_ > 0);
  }

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d", &input_isbigendian_);
  assert(ifail == 1);
  assert(input_isbigendian_ == 0 || input_isbigendian_ == 1);

  p =  fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &meta->npe);
  assert(ifail == 1);
  printf("Total number of processors %d\n", meta->npe);

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d",
		 &meta->pe[0], &meta->pe[1], &meta->pe[2]);
  assert(ifail == 3);
  printf("Decomposition is %d %d %d\n", meta->pe[0], meta->pe[1], meta->pe[2]);
  assert(meta->npe == meta->pe[0]*meta->pe[1]*meta->pe[2]);

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d",
		 &meta->ntotal[0], &meta->ntotal[1], &meta->ntotal[2]);
  assert(ifail == 3);
  printf("System size is %d %d %d\n",
	 meta->ntotal[0], meta->ntotal[1], meta->ntotal[2]);

  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%d", &meta->nplanes);
  assert(ifail == 1);
  assert(meta->nplanes >= 0);
  printf("Number of Lees Edwards planes %d\n", meta->nplanes);
  p =  fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);
  ifail = sscanf(tmp+ncharoffset, "%lf", &le_speed_);
  assert(ifail == 1);
  printf("Lees Edwards speed: %f\n", le_speed_);

  /* Number of I/O groups */
  p = fgets(tmp, FILENAME_MAX, fp_meta);
  assert(p);

  {
    int nio = atoi(tmp + ncharoffset);
    if (1 > nio || nio > 999) {
      printf("Invalid number of i/o groups %d\n", meta->nio);
      exit(-1);
    }
    else {
      meta->nio = nio;
    }
    printf("Number of I/O groups: %d\n", meta->nio);
  }

  /* I/O decomposition */
  p = fgets(tmp, FILENAME_MAX, fp_meta);
  if (p == NULL) printf("Not reached last line correctly\n");
  ifail = sscanf(tmp+ncharoffset, "%d %d %d\n",
		 io_size + 0, io_size + 1, io_size + 2);
  assert(ifail == 3);
  printf("I/O communicator topology: %d %d %d\n",
	 io_size[0], io_size[1], io_size[2]);

  fclose(fp_meta);

  /* Is this the velocity field? */

  if (nrec_ == 3 && strncmp(filename, "vel", 3) == 0) {
    is_velocity_ = 1;
    printf("\nThis is a velocity field\n");
  }

  return;
}

/****************************************************************************
 *
 *  read_data_file_name
 *
 *  This replies on a filename of the form stub-nnnnnn.001-001
 *  to identify the time step ('nnnnnnn').
 *
 ****************************************************************************/

int read_data_file_name(const char * filename) {

  const char * tmp = NULL;

  assert(filename);

  tmp = strchr(filename, '-');
  if (tmp) {
    int ntime = -1;
    int ns = sscanf(tmp+1, "%d.", &ntime);
    if (ns < 1) {
      printf("Could not determine time from %s\n", filename);
      assert(0);
    }
    else {
      return ntime;
    }
  }

  return -1;
}

/****************************************************************************
 *
 *  copy_data
 *
 *  Copy the local data into the global array in the correct place.
 *
 ****************************************************************************/

int copy_data(metadata_v1_t * meta, double * datalocal, double * datasection) {

  int ic, jc, kc, ic_g, jc_g, kc_g;
  int index_l, index_s, nr;

  /* Sweep through the local sites for this pe */

  for (ic = 1; ic <= meta->nlocal[0]; ic++) {
    ic_g = meta->offset[0] + ic;
    for (jc = 1; jc <= meta->nlocal[1]; jc++) {
      jc_g = meta->offset[1] + jc;
      for (kc = 1; kc <= meta->nlocal[2]; kc++) {
	kc_g = meta->offset[2] + kc;
	index_s = nrec_*site_index(ic_g, jc_g, kc_g, ntargets);

	for (nr = 0; nr < nrec_; nr++) {
	  index_l = nrec_*site_index(ic, jc, kc, meta->nlocal);
	  *(datasection + index_s + nr) = *(datalocal + index_l + nr);
	}
      }
    }
  }

  return 0;
}

/****************************************************************************
 *
 *  read_data
 *
 *  Read data block relevant for one pe
 *
 ****************************************************************************/

void read_data(FILE * fp_data, metadata_v1_t * meta, double * data) {

  int ic, jc, kc, index, nr;
  int nread;
  double phi;
  double revphi;

  assert(meta);

  if (input_binary_) {
    for (ic = 1; ic <= meta->nlocal[0]; ic++) {
      for (jc = 1; jc <= meta->nlocal[1]; jc++) {
	for (kc = 1; kc <= meta->nlocal[2]; kc++) {
	  index = site_index(ic, jc, kc, meta->nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    nread = fread(&phi, sizeof(double), 1, fp_data);
	    if (nread != 1) printf("File corrupted!\n");
	    if(reverse_byte_order_){
	       revphi = reverse_byte_order_double((char *) &phi); 
	       phi = revphi;
	    }
	    *(data + nrec_*index + nr) = phi;
	  }
	}
      }
    }
  }
  else {
    for (ic = 1; ic <= meta->nlocal[0]; ic++) {
      for (jc = 1; jc <= meta->nlocal[1]; jc++) {
	for (kc = 1; kc <= meta->nlocal[2]; kc++) {
	  index = site_index(ic, jc, kc, meta->nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    nread = fscanf(fp_data, "%le", data + nrec_*index + nr);
	    if (nread != 1) printf("File corrupted!\n");
	  }
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  write_data
 *
 *  In the current format.
 *
 ****************************************************************************/

int write_data(FILE * fp_data, int n[3], int nrec0, int nrec, double * data) {

  if (output_binary_) {
    write_data_binary(fp_data, n, nrec0, nrec, data);
  }
  else {
    write_data_ascii(fp_data, n, nrec0, nrec, data);
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_data_cmf
 *
 *  Column major format.
 *
 *****************************************************************************/

int write_data_cmf(FILE * fp_data, int n[3], int nr0, int nr, double * data) {

  if (output_binary_) {
    write_data_binary_cmf(fp_data, n, nr0, nr, data);
  }
  else {
    write_data_ascii_cmf(fp_data, n, nr0, nr, data);
  }

  return 0;
}

/****************************************************************************
 *
 *  le_displacement
 *
 ****************************************************************************/

int le_displacement(int ntotal[3], int ic, int jc) {

    int dj;

    dj = jc + le_displacements_[ic-1];

    dj = dj % ntotal[1];
    if (dj > ntargets[1]) dj -= ntargets[1];
    if (dj < 1) dj += ntargets[1];

    return dj;
}

/****************************************************************************
 *
 *  le_set_displacements
 *
 ****************************************************************************/

void le_set_displacements(metadata_v1_t * meta) {

  int ic;
  int di;
  double dy;
  double duy;

  for (ic = 0; ic < meta->ntotal[0]; ic++) {
    le_displacements_[ic] = 0.0;
    le_duy_[ic] = 0.0;
  }

  if (meta->nplanes > 0) {

    di = meta->ntotal[0] / meta->nplanes;
    dy = -(meta->nplanes/2.0)*le_displace_;
    duy = -(meta->nplanes/2.0)*le_speed_;

    /* Fist half block */
    for (ic = 1; ic <= di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
    }

    dy += le_displace_;
    duy += le_speed_;
    for (ic = di/2 + 1; ic <= meta->ntotal[0] - di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
      if ( (ic - di/2) % di == 0 ) {
	dy += le_displace_;
	duy += le_speed_;
      }
    }

    /* Last half block */
    for (ic = meta->ntotal[0] - di/2 + 1; ic <= meta->ntotal[0]; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
    }
  }

  return;
}

/****************************************************************************
 *
 *  site_index
 *
 ****************************************************************************/

int site_index(int ic, int jc, int kc, const int n[3]) {

  int index;

  index = (n[1]*n[2]*(ic-1) + n[2]*(jc-1) + kc-1);

  return index;
}


/*****************************************************************************
 *
 *  le_unroll
 *
 *  Unroll the data in the presence of planes.
 *  This is always done relative to the middle of the system (as defined
 *  in le_displacements_[]).
 *
 *  If this is the velocity field, we need to make a correction to u_y
 *  to allow for the motion of the planes.
 *
 *****************************************************************************/

void le_unroll(metadata_v1_t * meta, double * data) {

  int ic, jc, kc, n;
  int j0, j1, j2, j3, jdy;
  size_t ntmp;
  double * buffer;
  double dy, fr;
  double du[3];

  assert(meta);

  /* Allocate the temporary buffer */

  ntmp = (size_t) nrec_*ntargets[1]*ntargets[2];
  buffer = (double *) calloc(ntmp, sizeof(double));
  if (buffer == NULL) {
    printf("malloc(buffer) failed\n");
    exit(-1);
  }

  du[0] = 0.0;
  du[1] = 0.0;
  du[2] = 0.0;

  for (ic = 1; ic <= ntargets[0]; ic++) {
    dy = le_displacements_[ic-1];
    jdy = floor(dy);
    fr = 1.0 - (dy - jdy);
    if (is_velocity_) du[1] = le_duy_[ic-1];

    for (jc = 1; jc <= ntargets[1]; jc++) {
      j0 = 1 + (jc - jdy - 3 + 1000*meta->ntotal[1]) % meta->ntotal[1];
      j1 = 1 + j0 % meta->ntotal[1];
      j2 = 1 + j1 % meta->ntotal[1];
      j3 = 1 + j2 % meta->ntotal[1];

      for (kc = 1; kc <= ntargets[2]; kc++) {
	for (n = 0; n < nrec_; n++) {
	  buffer[nrec_*site_index(1,jc,kc,ntargets) + n] =
	    - (1.0/6.0)*fr*(fr-1.0)*(fr-2.0)
	               *data[nrec_*site_index(ic,j0,kc,ntargets) + n]
	    + 0.5*(fr*fr-1.0)*(fr-2.0)
	         *data[nrec_*site_index(ic,j1,kc,ntargets) + n]
	    - 0.5*fr*(fr+1.0)*(fr-2.0)
	         *data[nrec_*site_index(ic,j2,kc,ntargets) + n]
	    + (1.0/6.0)*fr*(fr*fr-1.0)
	               *data[nrec_*site_index(ic,j3,kc,ntargets) + n];
	}
      }
    }
    /* Put the whole buffer plane back in place */

    for (jc = 1; jc <= ntargets[1]; jc++) {
      for (kc = 1; kc <= ntargets[2]; kc++) {
	for (n = 0; n < nrec_; n++) {
	  data[nrec_*site_index(ic,jc,kc,ntargets) + n] =
	    buffer[nrec_*site_index(1,jc,kc,ntargets) + n];
	  if (n < 3) {
	    data[nrec_*site_index(ic,jc,kc,ntargets) + n] += du[n];
	  }
	}
      }
    }

  }

  free(buffer);

  return;
}

/*****************************************************************************
 *
 *  write_vtk_header
 *
 *  Suitable for an ascii file; to be followed by actual data.
 *
 *****************************************************************************/

int write_vtk_header(FILE * fp, int nrec, int ndim[3], const char * des,
		     vtk_enum_t vtk) {

  if (output_vtk_ == 1) write_vtk_header_points(fp, nrec, ndim, des, vtk);
  if (output_vtk_ == 2) write_vtk_header_rectilinear(fp, nrec, ndim, des, vtk);

  return 0;
}

int write_vtk_header_points(FILE * fp, int nrec, int ndim[3],
			    const char * descript,
			    vtk_enum_t vtk) {

  assert(fp);

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "Generated by ludwig extract.c\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET STRUCTURED_POINTS\n");
  fprintf(fp, "DIMENSIONS %d %d %d\n", ndim[0], ndim[1], ndim[2]);
  fprintf(fp, "ORIGIN %d %d %d\n", 0, 0, 0);
  fprintf(fp, "SPACING %d %d %d\n", 1, 1, 1);
  fprintf(fp, "POINT_DATA %d\n", ndim[0]*ndim[1]*ndim[2]);
  if (vtk == VTK_SCALARS) {
    fprintf(fp, "SCALARS %s float %d\n", descript, nrec);
    fprintf(fp, "LOOKUP_TABLE default\n");
  }
  if (vtk == VTK_VECTORS) {
    fprintf(fp, "VECTORS %s float\n", descript);
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_vtk_header_rectilinear
 *
 *  Suitable for an ascii file; to be followed by actual data.
 *
 *  This one is useful is cell-centre data is the best view;
 *  we set the coordinates of the cell edges in each dimension.
 *
 *****************************************************************************/

int write_vtk_header_rectilinear(FILE * fp, int nrec, int ndim[3],
				 const char * descript,
				 vtk_enum_t vtk) {

  assert(fp);

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "Generated by ludwig extract.c\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET RECTILINEAR_GRID\n");
  fprintf(fp, "DIMENSIONS %d %d %d\n", ndim[0]+1, ndim[1]+1, ndim[2]+1);

  fprintf(fp, "X_COORDINATES %d float\n", ndim[0]+1);
  for (int ic = 0; ic <= ndim[0]; ic++) {
    fprintf(fp, " %5.1f", 0.5+ic);
  }
  fprintf(fp, "\n");

  fprintf(fp, "Y_COORDINATES %d float\n", ndim[1]+1);
  for (int jc = 0; jc <= ndim[1]; jc++) {
    fprintf(fp, " %5.1f", 0.5+jc);
  }
  fprintf(fp, "\n");

  fprintf(fp, "Z_COORDINATES %d float\n", ndim[2]+1);
  for (int kc = 0; kc <= ndim[2]; kc++) {
    fprintf(fp, " %5.1f", 0.5+kc);
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELL_DATA %d", ndim[0]*ndim[1]*ndim[2]);

  if (vtk == VTK_SCALARS) {
    fprintf(fp, "SCALARS %s float %d\n", descript, nrec);
    fprintf(fp, "LOOKUP_TABLE default\n");
  }
  if (vtk == VTK_VECTORS) {
    fprintf(fp, "VECTORS %s float\n", descript);
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_qab_vtk
 *
 *  For Q_ab in vtk format, three separate files are required:
 *    scalar order parameter  (scalar)
 *    director                (vector)
 *    biaxial order parameter (scalar)
 *
 *****************************************************************************/

int write_qab_vtk(int ntime, int ntarget[3], double * datasection) {

  char io_data[FILENAME_MAX];
  FILE * fp_data = NULL;

  assert(datasection);
  assert(nrec_ == 5);

  /* Scalar order */
  if (output_lcs_) {
    sprintf(io_data, "lcs-%8.8d.vtk", ntime);
    fp_data = util_fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed scalar order with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntarget, "Q_ab_scalar_order", VTK_SCALARS);
    if (output_cmf_ == 0) write_data(fp_data, ntarget, 0, 1, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntarget, 0, 1, datasection);
    fclose(fp_data);
  }

  /* Director */
  fp_data = NULL;

  if (output_lcd_) {
    sprintf(io_data, "lcd-%8.8d.vtk", ntime);
    fp_data = util_fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed director with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntarget, "Q_ab_director", VTK_VECTORS);
    if (output_cmf_ == 0) write_data(fp_data, ntarget, 1, 3, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntarget, 1, 3, datasection);
    fclose(fp_data);
  }

  /* Biaxial order */
  fp_data = NULL;

  if (output_lcx_) {
    sprintf(io_data, "lcb-%8.8d.vtk", ntime);
    fp_data = util_fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed biaxial order with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntarget, "Q_ab_biaxial_order", VTK_SCALARS);
    if (output_cmf_ == 0) write_data(fp_data, ntarget, 4, 1, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntarget, 4, 1, datasection);
    fclose(fp_data);
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_data_binary
 *
 *  For data of section size n[3].
 *  Records [nrec0:nrec0+nrec] are selected from total nrec_.
 *
 *****************************************************************************/

int write_data_binary(FILE * fp, int n[3], int nrec0, int nrec, double * data) {

  int ic, jc, kc;
  int index, nr;

  assert(fp);
  assert((nrec0 + nrec) <= nrec_);

  for (ic = 1; ic <= n[0]; ic++) {
    for (jc = 1; jc <= n[1]; jc++) {
      for (kc = 1; kc <= n[2]; kc++) {

	index = site_index(ic, jc, kc, n);
	for (nr = nrec0; nr < (nrec0 + nrec); nr++) {
	  fwrite(data + nrec_*index + nr, sizeof(double), 1, fp);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_data_ascii
 *
 *  For data of size n[3].
 *  Records nrec0 .. nrec0+nrec from global nrec_ are selected.
 *
 *****************************************************************************/

int write_data_ascii(FILE * fp, int n[3], int nrec0, int nrec, double * data) {

  int ic, jc, kc;
  int index, nr;

  assert(fp);
  assert((nrec0 + nrec) <= nrec_);

  for (ic = 1; ic <= n[0]; ic++) {
    for (jc = 1; jc <= n[1]; jc++) {
      for (kc = 1; kc <= n[2]; kc++) {

	index = site_index(ic, jc, kc, n);
	if (output_index_) {
	  fprintf(fp, "%4d %4d %4d ", ic, jc, kc);
	}

	for (nr = nrec0; nr < (nrec0 + nrec - 1); nr++) {
	  fprintf(fp, "%13.6e ", *(data + nrec_*index + nr));
	}
	/* Last one has a new line (not space) */
	nr = nrec0 + nrec - 1;
	fprintf(fp, "%13.6e\n", *(data + nrec_*index + nr));
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_data_binary_cmf
 *
 *****************************************************************************/

int write_data_binary_cmf(FILE * fp, int n[3], int nrec0, int nrec,
			  double * data) {
  int ic, jc, kc;
  int index, nr;

  assert(fp);
  assert((nrec0 + nrec) <= nrec_);

  for (kc = 1; kc <= n[2]; kc++) {
    for (jc = 1; jc <= n[1]; jc++) {
      for (ic = 1; ic <= n[0]; ic++) {

	index = site_index(ic, jc, kc, n);
	for (nr = nrec0; nr < (nrec0 + nrec); nr++) {
	  fwrite(data + nrec_*index + nr, sizeof(double), 1, fp);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  write_data_ascii_cmf
 *
 *****************************************************************************/

int write_data_ascii_cmf(FILE * fp, int n[3], int nrec0, int nrec,
			 double * data) {
  int ic, jc, kc;
  int index, nr;

  assert(fp);
  assert((nrec0 + nrec) <= nrec_);

  for (kc = 1; kc <= n[2]; kc++) {
    for (jc = 1; jc <= n[1]; jc++) {
      for (ic = 1; ic <= n[0]; ic++) {
	index = site_index(ic, jc, kc, n);
      
	if (output_index_) {
	  fprintf(fp, "%4d %4d %4d ", ic, jc, kc);
	}

	for (nr = nrec0; nr < (nrec0 + nrec - 1); nr++) {
	  fprintf(fp, "%13.6e ", *(data + nrec_*index + nr));
	}
	nr = nrec0 + nrec - 1;
	fprintf(fp, "%13.6e\n", *(data + nrec_*index + nr));
      }
    }
  }

  return 0;
}

/****************************************************************************
 *
 *  lc_transform_q5
 *
 *  This routine diagonalises the Q tensor [q_xx, q_xy, q_xz, q_yy, q_yz]
 *  and replaces it with [s, n_x, n_y, n_z, b], where s is the scalar
 *  order parameter, (n_x, n_y, n_z) is the director, and b is the
 *  biaxial order parameter.
 *
 *  nlocal is usually the entire system size.
 *
 *****************************************************************************/

int lc_transform_q5(int * nlocal, double * datasection) {

  int ic, jc, kc, nr, index;
  int ifail = 0;
  double q[3][3], qs[5];

  assert(nrec_ == 5);
  assert(datasection);

  for (ic = 1; ic <= nlocal[0]; ic++) {
    for (jc = 1; jc <= nlocal[1]; jc++) {
      for (kc = 1; kc <= nlocal[2]; kc++) {

	index = nrec_*site_index(ic, jc, kc, nlocal);

	q[0][0] = *(datasection + index + 0);
	q[1][0] = *(datasection + index + 1);
	q[2][0] = *(datasection + index + 2);
	q[0][1] = q[1][0];
	q[1][1] = *(datasection + index + 3);
	q[2][1] = *(datasection + index + 4);
	q[0][2] = q[2][0];
	q[1][2] = q[2][1];
	q[2][2] = 0.0 - q[0][0] - q[1][1];

	ifail += lc_compute_scalar_ops(q, qs);

	for (nr = 0; nr < nrec_; nr++) {
	  *(datasection + index + nr) = qs[nr];
	}
	if (ifail != 0) {
	  printf("! fail diagonalisation at %3d %3d %3d\n", ic, jc, kc);
	}
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  compute_scalar_ops   [A copy from src/lc_blue_phase]
 *
 *  For symmetric traceless q[3][3], return the associated scalar
 *  order parameter, biaxial order parameter and director:
 *
 *  qs[0]  scalar order parameter: largest eigenvalue
 *  qs[1]  director[X] (associated eigenvector)
 *  qs[2]  director[Y]
 *  qs[3]  director[Z]
 *  qs[4]  biaxial order parameter b = sqrt(1 - 6 (Tr(QQQ))^2 / Tr(QQ)^3)
 *         related to the two largest eigenvalues...
 *
 *  If we write Q = ((s, 0, 0), (0, t, 0), (0, 0, -s -t)) then
 *
 *    Tr(QQ)  = s^2 + t^2 + (s + t)^2
 *    Tr(QQQ) = 3 s t (s + t)
 *
 *  If no diagonalisation is possible, all the results are set to zero.
 *
 *****************************************************************************/

int lc_compute_scalar_ops(double q[3][3], double qs[5]) {

  int ifail;
  double eigenvalue[3];
  double eigenvector[3][3];
  double s, t;
  double q2, q3;

  ifail = util_jacobi_sort(q, eigenvalue, eigenvector);

  qs[0] = 0.0; qs[1] = 0.0; qs[2] = 0.0; qs[3] = 0.0; qs[4] = 0.0;

  if (ifail == 0) {

    double q4 = 0.0;

    qs[0] = eigenvalue[0];
    qs[1] = eigenvector[0][0];
    qs[2] = eigenvector[1][0];
    qs[3] = eigenvector[2][0];

    s = eigenvalue[0];
    t = eigenvalue[1];

    q2 = s*s + t*t + (s + t)*(s + t);
    q3 = 3.0*s*t*(s + t);

    /* Note the value here can drip just below zero by about DBL_EPSILON */
    /* So just set to zero to prevent an NaN */
    q4 = 1.0 - 6.0*q3*q3 / (q2*q2*q2);
    if (q4 < 0.0) q4 = 0.0;
    qs[4] = sqrt(q4);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  file_stub_valid
 *
 *  Return a recognised name, or NULL if the input is not recognised.
 *
 *****************************************************************************/

const char * file_stub_valid(const char * input) {

  const char * output = NULL;

  if (strncmp(input, "phi", 3) == 0) {
    output = "phi"; /* scalar order parameter(s) */
  }
  else if (strncmp(input, "psi", 3) == 0) {
    output = "psi"; /* charge */
  }
  else if (strncmp(input, "dist", 4) == 0) {
    output = "dist";  /* distributions */
  }
  else if (strncmp(input, "rho", 3) == 0) {
    output = "rho";   /* density */
  }
  else if (strncmp(input, "vel", 3) == 0) {
    output = "vel";   /* velocity field */
  }
  else if (strncmp(input, "p", 1) == 0) {
    output = "p";   /* vector order parameter */
  }
  else if (strncmp(input, "q", 1) == 0) {
    output = "q";   /* tensor order parameter */
  }

  return output;
}

/*****************************************************************************
 *
 *  file_get_file_nfile
 *
 *  Metadata encoded in a set of parallel, decomposed, filenames e.g.,
 *    phi-metadata.002-001
 *    phi-metadata.002-002
 *  would return nfile = 2.
 *
 *****************************************************************************/

int file_get_file_nfile(const char * filename) {

  int nfile = 0;
  char dot = '.';
  const char * str1 = strchr(filename, dot); /* First "." */

  if (str1 != NULL) {
    char buf[BUFSIZ] = {0};
    strncpy(buf, str1 + 1, 3);
    nfile = atoi(buf);
  }

  return nfile;
}

/*****************************************************************************
 *
 *  file_get_file_index
 *
 *  Likewise, returns the index (see file_get_file_nfile above).
 *
 *****************************************************************************/

int file_get_file_index(const char * filename) {

  int ifile = 0;
  char dash = '-';
  const char * str1 = strrchr(filename, dash); /* Last "-" */

  if (str1 != NULL) {
    char buf[BUFSIZ] = {0};
    strncpy(buf, str1 + 1, 3);
    ifile = atoi(buf);
  }

  return ifile;
}


/*****************************************************************************
 *
 *  extract_read_single_file
 *
 *****************************************************************************/

int io_metadata_nrecord(const io_metadata_t * meta) {

  /* This is still not very satisfactory. Need no,. of records in metadata */
  int nrecord = meta->element.count;
  if (meta->element.datatype == MPI_CHAR) nrecord /= 23;
  nrec_ = nrecord;

  return nrecord;
}

int extract_read_single_file(io_metadata_t * meta, const char * stub,
			     int itime, double * datatotal) {

  int nrecord = io_metadata_nrecord(meta);
  char filename[BUFSIZ] = {0};
  FILE * fp = NULL;

  io_subfile_name(&meta->subfile, stub, itime, filename, BUFSIZ);

  fp = util_fopen(filename, "r+b");
  if (fp == NULL) printf("fopen(%s) failed\n", filename);

  for (int ic = 1; ic <= meta->subfile.sizes[X]; ic++) {
    for (int jc = 1; jc <= meta->subfile.sizes[Y]; jc++) {
      for (int kc = 1; kc <= meta->subfile.sizes[Z]; kc++) {

	int icd = meta->subfile.offset[X] + ic;
	int jcd = meta->subfile.offset[Y] + jc;
	int kcd = meta->subfile.offset[Z] + kc;
	int indexd = site_index(icd, jcd, kcd, meta->cs->param->ntotal);

	for (int nr = 0; nr < nrecord; nr++) {
	  double datum = 0.0;
	  int nread = 0;
	  if (meta->options.iorformat == IO_RECORD_BINARY) {
	    nread = fread(&datum, sizeof(double), 1, fp);
	  }
	  else {
	    nread = fscanf(fp, "%le", &datum);
	  }
	  if (nread != 1) printf("File out of records!!\n");

	  /* Place at correct offset in the full array */
	  *(datatotal + nrecord*indexd + nr) = datum;
	}

      }
    }
  }

  fclose(fp);

  return 0;
}

/*****************************************************************************
 *
 *  extract_unroll
 *
 *****************************************************************************/

int le_block_id(int nplanes, int nx, int ic) {

  /* Could take local ic and and offset here */
  /* Halo points should be in the outer most block (always) */

  assert(nplanes >= 0);
  assert(nx > 0);
  assert(1 <= ic && ic <= nx);

  int bid = 0;

  if (nplanes > 0) {
    int s   = nx/nplanes;        /* Separation */ 
    int ic0 = nx/2;              /* Ref. is centre */

    if (nplanes % 2 == 0) {
      /* blocks are ... -2 | -1 | 0 | +1 | +2 ... */
      if (ic >  ic0) bid = (+s/2 + ic - ic0 - 1)/s;
      if (ic <= ic0) bid = (-s/2 + ic - ic0    )/s;
    }
    else {
      /* blocks are ... -2 | -1 | +1 | +2 ... (no "block zero") */
      if (ic >  ic0) bid =  1 + (ic - ic0 - 1)/s;
      if (ic <= ic0) bid = -1 + (ic - ic0    )/s;
    }
  }

  return bid;
}

double le_block_displacement_uyt(const cs_t * cs, int ic, int itime) {

  int nplanes = cs->leopts.nplanes;
  double uyt = 0.0;

  if (nplanes > 0) {
    int nx = cs->param->ntotal[X];
    double uy = cs->leopts.uy;

    if (nplanes % 2 == 0) {
      uyt = uy*itime*le_block_id(nplanes, nx, ic);
    }
    else {
      int bid = le_block_id(nplanes, nx, ic);
      if (bid <  0) uyt = uy*itime*(0.5 + bid);
      if (bid >= 0) uyt = uy*itime*(bid - 0.5);
    }
  }

  return uyt;
}

/* A function of time only for non-steady shear. */

double le_block_uy(cs_t * cs, int ic, int itime) {

  int nplanes = cs->leopts.nplanes;
  int nx = cs->param->ntotal[X];
  double uy = 0.0;

  if (nplanes > 0) {
    int bid = le_block_id(nplanes, nx, ic);
    if (nplanes % 2 == 0) {
     uy = cs->leopts.uy*bid;
    }
    else {
      if (bid > 0) uy = cs->leopts.uy*(bid - 0.5);
      if (bid < 0) uy = cs->leopts.uy*(bid + 0.5);
    }
  }

  return uy;
}

/*****************************************************************************
 *
 *  extract_unroll
 *
 *  Unroll the time-dependent LE displacements.
 *
 *  If it's the velocity, there's a correction to u_y to account for
 *  the relative block motion.
 *
 *****************************************************************************/

int extract_unroll(const io_metadata_t * meta, double * data, int itime,
		   int is_vel) {

  int nrecord = io_metadata_nrecord(meta);
  int ntotal[3] = {0};
  double * buffer = NULL;
  double du[BUFSIZ] = {0};

  cs_ntotal(meta->cs, ntotal);

  {
    /* Temporary for y-z plane (all reocrds) */
    int nsz = nrecord*ntotal[Y]*ntotal[Z];
    buffer = (double *) malloc(nsz*sizeof(double));
    if (buffer == NULL) printf("malloc(buffer) failed!\n");
  }

  for (int ic = 1; ic <= ntotal[X]; ic++) {

    double dy = le_block_displacement_uyt(meta->cs, ic, itime);
    int   jdy = floor(dy);
    double fr = 1.0 - (dy - jdy);
    if (is_vel) du[Y] = le_block_uy(meta->cs, ic, itime);
    printf("time %6d ic %3d  uyt %7.3f dv %7.4f %3d\n" , itime, ic, dy,
	   le_block_uy(meta->cs, ic, itime),
	   le_block_id(meta->cs->leopts.nplanes, ntotal[X], ic));

    for (int jc = 1; jc <= ntotal[Y]; jc++) {
      int j0 = 1 + (jc - jdy - 3 + 1000*ntotal[Y]) % ntotal[Y];
      int j1 = 1 + j0 % ntotal[Y];
      int j2 = 1 + j1 % ntotal[Y];
      int j3 = 1 + j2 % ntotal[Y];

      for (int kc = 1; kc <= ntotal[Z]; kc++) {
	for (int n = 0; n < nrecord; n++) {
	  int index0 = site_index(ic, j0, kc, ntotal);
	  int index1 = site_index(ic, j1, kc, ntotal);
	  int index2 = site_index(ic, j2, kc, ntotal);
	  int index3 = site_index(ic, j3, kc, ntotal);
	  buffer[nrecord*site_index(1,jc,kc,ntotal) + n] =
	    - (1.0/6.0)*fr*(fr-1.0)*(fr-2.0)*data[nrecord*index0 + n]
	    + 0.5*(fr*fr-1.0)*(fr-2.0)*data[nrecord*index1 + n]
	    - 0.5*fr*(fr+1.0)*(fr-2.0)*data[nrecord*index2 + n]
	    + (1.0/6.0)*fr*(fr*fr-1.0)*data[nrecord*index3 + n];
	}
      }
    }

    /* Put the whole buffer plane back in place */

    for (int jc = 1; jc <= ntotal[Y]; jc++) {
      for (int kc = 1; kc <= ntotal[Z]; kc++) {
	int index  = site_index(ic, jc, kc, ntotal);
	int index1 = site_index(1,  jc, kc, ntotal);
	for (int n = 0; n < nrecord; n++) {
	  data[nrecord*index + n] = buffer[nrecord*index1 + n] + du[n];
	}
      }
    }
    /* Next ic */
  }

  free(buffer);

  return 0;
}

/*****************************************************************************
 *
 *  extract_process_and_output
 *
 *****************************************************************************/

int extract_process_and_output(const char * stub, int ntime,
			       io_metadata_t * meta) {

  int nrecord = -1;                   /* Number of records per site (nf) */
  double * datasection = NULL;        /* field data */
  FILE * fp_output = NULL;            /* Output file */

  assert(stub);
  assert(0 <= ntime && ntime < 1000*1000*1000);

  /* Records */

  nrecord = io_metadata_nrecord(meta);
  assert(nrecord > 0);

  /* No. sites in the target section (always original at moment) */

  cs_ntotal(meta->cs, ntargets);

  {
    size_t n = nrecord*ntargets[0]*ntargets[1]*ntargets[2];
    datasection = (double *) calloc(n, sizeof(double));
    if (datasection == NULL) printf("calloc(datasection) failed\n");
  }

  extract_read_single_file(meta, stub, ntime, datasection);

  /* Unroll the data if Lees Edwards planes are present */

  if (meta->cs->leopts.nplanes > 0) {
    int is_vel = (strcmp(stub, "vel") == 0);
    printf("Unrolling LE planes from centre (displacement %f)\n", -999.999);
    if (is_vel) printf("Data is velocity field (nrecords = %d)\n", nrecord);
    extract_unroll(meta, datasection, ntime, is_vel);
  }

  if (nrecord == 5 && strncmp(stub, "q", 1) == 0) {

    if (output_vtk_ == 1 || output_vtk_ == 2) {
      /* We mandate this is the transformed Q_ab */
      lc_transform_q5(ntargets, datasection);
      write_qab_vtk(ntime, ntargets, datasection);
    }
    else {
      char io_data[BUFSIZ] = {0};
      sprintf(io_data, "q-%9.9d", ntime);
      fp_output = util_fopen(io_data, "w+b");
      if (fp_output == NULL) {
	printf("fopen(%s) failed\n", io_data);
	exit(-1);
      }

      /* Here, we could respect the -d, -s, -x options. */
      /* But at the moment, always 5 components ... */
      if (output_q_raw_) {
	printf("Writing raw q to %s\n", io_data);
      }
      else {
	printf("Writing computed scalar q etc: %s\n", io_data);
	lc_transform_q5(ntargets, datasection);
      }

      if (output_cmf_ == 0) write_data(fp_output, ntargets, 0, 5, datasection);
      if (output_cmf_ == 1) write_data_cmf(fp_output, ntargets, 0, 5, datasection);

      fclose(fp_output);
    }
  }
  else {
    /* A direct input / output */

    /* Write a single file with the final section */
    char io_data[BUFSIZ] = {0};
    snprintf(io_data, sizeof(io_data), "%s-%9.9d", stub, ntime);

    if (output_vtk_ == 1 || output_vtk_ == 2) {

      strcat(io_data, ".vtk");
      fp_output = util_fopen(io_data, "w+b");
      if (fp_output == NULL) printf("fopen(%s) failed\n", io_data);
      printf("\nWriting result to %s\n", io_data);

      if (nrecord == 3 && strncmp(stub, "vel", 3) == 0) {
	write_vtk_header(fp_output, nrecord, ntargets, "velocity_field",
			 VTK_VECTORS);
      }
      else if (nrecord == 1 && strncmp(stub, "phi", 3) == 0) {
	write_vtk_header(fp_output, nrecord, ntargets, "composition",
			 VTK_SCALARS);
      }
      else {
	/* Assume scalars */
	write_vtk_header(fp_output, nrecord, ntargets, stub, VTK_SCALARS);
      }

    }
    else {

      fp_output = util_fopen(io_data, "w+b");
      if (fp_output == NULL) printf("fopen(%s) failed\n", io_data);
      printf("\nWriting result to %s\n", io_data);

    }

    if (output_cmf_ == 0) {
      write_data(fp_output, ntargets, 0, nrecord, datasection);
    }
    if (output_cmf_ == 1) {
      write_data_cmf(fp_output, ntargets, 0, nrecord, datasection);
    }

    fclose(fp_output);
  }

  free(datasection);

  return 0;
}
