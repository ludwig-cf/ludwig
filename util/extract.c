/****************************************************************************
 *
 *  extract.c
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
 *  -k   Request VTK header (none by default)
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
 *  should produce a.out
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henirch  (ohenrich@strath.ac.uk)
 *
 *  (c) 2011-2018 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/util.h"

const int version = 2;        /* Meta data version */
                              /* 1 = older output files */
                              /* 2 = current processor independent per file */

typedef enum vtk_enum {VTK_SCALARS, VTK_VECTORS} vtk_enum_t;

int nlocal[3];
int rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz;

int ntotal[3];
int ntargets[3]; /* Output target */
int io_size[3];                /* I/O decomposition */
int pe_[3];
int nplanes_ = 0;
int nio_;
int nrec_ = 1;
int input_isbigendian_ = -1;   /* May need to deal with endianness */
int reverse_byte_order_ = 0;   /* Switch for bigendian input */
int input_binary_ = 1;         /* Switch for format of input */
int output_binary_ = 1;        /* Switch for format of final output */
int is_velocity_ = 0;          /* Switch to identify velocity field */
int output_index_ = 0;         /* For ASCII output, include (i,j,k) indices */
int output_vtk_ = 0;           /* Write ASCII VTK header */

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

char stub_[FILENAME_MAX];

int extract_driver(const char * filename, int version);
int read_version1(int ntime, int nlocal[3], double * datasection);
int read_version2(int ntime, int nlocal[3], double * datasection);

void read_meta_data_file(const char *);
int  read_data_file_name(const char *);

int write_data(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_cmf(FILE * fp, int n[3], int nr0, int nr, double * data);
int write_data_ascii(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_ascii_cmf(FILE * fp, int n[3], int nrec0, int nrec, double *);
int write_data_binary(FILE * fp, int n[3], int nrec0, int nrec, double * data);
int write_data_binary_cmf(FILE * fp, int n[3], int nrec0, int nrec, double *);

int site_index(int, int, int, const int *);
void read_data(FILE *, int *, double *);
int write_vtk_header(FILE * fp, int nrec, int ndim[3], const char * descript,
		     vtk_enum_t vtk);
int write_qab_vtk(int ntime, int ntargets[3], double * datasection);

int copy_data(double *, double *);
int le_displacement(int, int);
void le_set_displacements(void);
void le_unroll(double *);

int lc_transform_q5(int * nlocal, double * datasection);
int lc_compute_scalar_ops(double q[3][3], double qs[5]);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  size_t optind;

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

  if (optind > argc-2) {
    printf("Usage: %s [-abk] meta-file data-file\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  read_meta_data_file(argv[optind]);

  extract_driver(argv[optind+1], version);

  return 0;
}

/*****************************************************************************
 *
 *  extract_driver
 *
 *****************************************************************************/

int extract_driver(const char * filename, int version) {

  int ntime;
  int i, n;

  double * datasection;
  char io_data[FILENAME_MAX];

  FILE * fp_data;

  ntime = read_data_file_name(filename);

  /* Work out parallel local file size */

  assert(io_size[0] == 1);

  switch (version) {
  case 1:
    for (i = 0; i < 3; i++) {
      nlocal[i] = ntotal[i]/pe_[i];
    }
    break;
  case 2:
    for (i = 0; i < 3; i++) {
      nlocal[i] = ntotal[i]/io_size[i];
    }
    break;
  default:
    printf("Invalid version %d\n", version);
  }
    
  /* No. sites in the target section (always original at moment) */

  for (i = 0; i < 3; i++) {
    ntargets[i] = ntotal[i];
  }
 
  n = nrec_*ntargets[0]*ntargets[1]*ntargets[2];
  datasection = (double *) calloc(n, sizeof(double));
  if (datasection == NULL) printf("calloc(datasection) failed\n");

  /* LE displacements as function of x */
  le_displace_ = le_speed_*(double) (ntime - le_t0_);
  le_displacements_ = (double *) malloc(ntotal[0]*sizeof(double));
  le_duy_ = (double *) malloc(ntotal[0]*sizeof(double));
  if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
  if (le_duy_ == NULL) printf("malloc(le_duy_) failed\n");
  le_set_displacements();

  /* Read data file or files */

  if (version == 1) read_version1(ntime, nlocal, datasection);
  if (version == 2) read_version2(ntime, nlocal, datasection);

  /* Unroll the data if Lees Edwards planes are present */

  if (nplanes_ > 0) {
    printf("Unrolling LE planes from centre (displacement %f)\n",
	   le_displace_);
    le_unroll(datasection);
  }

  if (nrec_ == 5 && strncmp(stub_, "q", 1) == 0) {

    if (output_vtk_ == 1) {
      /* We mandate this is the transformed Q_ab */
      lc_transform_q5(ntargets, datasection);
      write_qab_vtk(ntime, ntargets, datasection);
    }
    else {
      sprintf(io_data, "q-%8.8d", ntime);
      fp_data = fopen(io_data, "w+b");
      if (fp_data == NULL) {
	printf("fopen(%s) failed\n", io_data);
	exit(-1);
      }

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
  else if (nrec_ == 3 && strncmp(stub_, "fed", 3) == 0) {
    /* Require a more robust method to identify free energy densities */
    assert(0); /* Requires an update */
  }
  else {
    /* A direct input / output */

    /* Write a single file with the final section */

    sprintf(io_data, "%s-%8.8d", stub_, ntime);
    fp_data = fopen(io_data, "w+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    printf("\nWriting result to %s\n", io_data);

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

int read_version2(int ntime, int nlocal[3], double * datasection) {

  int n;
  char io_data[BUFSIZ];

  double * datalocal = NULL;
  FILE * fp_data = NULL;

  n = nrec_*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) {
    printf("calloc(datalocal) failed\n");
    exit(-1);
  }

  /* Main loop */

  for (n = 1; n <= nio_; n++) {

    /* Open the current data file */

    sprintf(io_data, "%s-%8.8d.%3.3d-%3.3d", stub_, ntime, nio_, n);
    printf("-> %s\n", io_data);

    fp_data = fopen(io_data, "r+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    /* Read data file based on offsets recorded in the metadata,
     * then copy this section to the global array */

    nlx = nlocal[0]; nly = nlocal[1]; nlz = nlocal[2];
    nox = 0; noy = 0; noz = 0;

    read_data(fp_data, nlocal, datalocal);
    copy_data(datalocal, datasection);

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

int read_version1(int ntime, int nlocal[3], double * datasection) {

  int ifail;
  int n;
  int p, pe_per_io;
  char io_metadata[BUFSIZ];
  char io_data[BUFSIZ];
  char line[BUFSIZ];

  double * datalocal = NULL;
  FILE * fp_metadata = NULL;
  FILE * fp_data = NULL;

  pe_per_io = pe_[0]*pe_[1]*pe_[2]/nio_;

  /* Allocate storage */

  n = nrec_*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) {
    printf("calloc(datalocal) failed\n");
    exit(-1);
  }

  /* Main loop */

  for (n = 1; n <= nio_; n++) {

    /* Open metadata file and skip 12 lines to get to the
     * decomposition inofmration */

    sprintf(io_metadata, "%s.%3.3d-%3.3d.meta", stub_, nio_, n);
    printf("Reading metadata file ... %s ", io_metadata);

    fp_metadata = fopen(io_metadata, "r");
    if (fp_metadata == NULL) printf("fopen(%s) failed\n", io_metadata);

    for (p = 0; p < 12; p++) {
      fgets(line, FILENAME_MAX, fp_metadata);
      printf("%s", line);
    }

    /* Open the current data file */

    sprintf(io_data, "%s-%8.8d.%3.3d-%3.3d", stub_, ntime, nio_, n);
    printf("-> %s\n", io_data);

    fp_data = fopen(io_data, "r+b");
    if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

    /* Read data file based on offsets recorded in the metadata,
     * then copy this section to the global array */

    for (p = 0; p < pe_per_io; p++) {

      ifail = fscanf(fp_metadata, "%d %d %d %d %d %d %d %d %d %d",
		     &rank, &cx, &cy, &cz, &nlx, &nly, &nlz, &nox, &noy, &noz);
      assert(ifail == 10);

      read_data(fp_data, nlocal, datalocal);
      copy_data(datalocal, datasection);
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

void read_meta_data_file(const char * filename) {

  int npe, nrbyte;
  int ifail;
  char tmp[FILENAME_MAX];
  FILE * fp_meta;
  const int ncharoffset = 33;

  fp_meta = fopen(filename, "r");
  if (fp_meta == NULL) {
    printf("fopen(%s) failed\n", filename);
    exit(-1);
  }

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%s\n", stub_);
  assert(ifail == 1);
  printf("Read stub: %s\n", stub_);
  fgets(tmp, FILENAME_MAX, fp_meta);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &nrbyte);
  assert(ifail == 1);
  printf("Record size (bytes): %d\n", nrbyte);
  if (input_binary_ == 1) {
    assert((nrbyte % 8) == 0);
    nrec_ = nrbyte / 8;
  }
  else {
    /* ASCCI */
    nrec_ = nrbyte/22;
  }

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d", &input_isbigendian_);
  assert(ifail == 1);
  assert(input_isbigendian_ == 0 || input_isbigendian_ == 1);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d\n", &npe);
  assert(ifail == 1);
  printf("Total number of processors %d\n", npe);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d", pe_, pe_+1, pe_+2);
  assert(ifail == 3);
  printf("Decomposition is %d %d %d\n", pe_[0], pe_[1], pe_[2]);
  assert(npe == pe_[0]*pe_[1]*pe_[2]);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d %d %d", ntotal, ntotal+1, ntotal+2);
  assert(ifail == 3);
  printf("System size is %d %d %d\n", ntotal[0], ntotal[1], ntotal[2]);

  fgets(tmp, FILENAME_MAX, fp_meta);
  ifail = sscanf(tmp+ncharoffset, "%d", &nplanes_);
  assert(ifail == 1);
  assert(nplanes_ >= 0);
  printf("Number of Lees Edwards planes %d\n", nplanes_);
  fgets(tmp, FILENAME_MAX, fp_meta);
  sscanf(tmp+ncharoffset, "%lf", &le_speed_);
  printf("Lees Edwards speed: %f\n", le_speed_);

  /* Number of I/O groups */
  fgets(tmp, FILENAME_MAX, fp_meta);
  sscanf(tmp+ncharoffset, "%d", &nio_);
  printf("Number of I/O groups: %d\n", nio_);
  /* I/O decomposition */
  fgets(tmp, FILENAME_MAX, fp_meta);
  sscanf(tmp+ncharoffset, "%d %d %d\n", io_size + 0, io_size + 1, io_size + 2);
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

  int ntime = -1;
  char * tmp;
  
  tmp = strchr(filename, '-');
  if (tmp) {
    sscanf(tmp+1, "%d.", &ntime);
  }

  assert (ntime >= 0);
  return ntime;
}

/****************************************************************************
 *
 *  copy_data
 *
 *  Copy the local data into the global array in the correct place.
 *
 ****************************************************************************/

int copy_data(double * datalocal, double * datasection) {

  int ic, jc, kc, ic_g, jc_g, kc_g;
  int index_l, index_s, nr;

  /* Sweep through the local sites for this pe */

  for (ic = 1; ic <= nlocal[0]; ic++) {
    ic_g = nox + ic;
    for (jc = 1; jc <= nlocal[1]; jc++) {
      jc_g = noy + jc;
      for (kc = 1; kc <= nlocal[2]; kc++) {
	kc_g = noz + kc;
	index_s = nrec_*site_index(ic_g, jc_g, kc_g, ntargets);

	for (nr = 0; nr < nrec_; nr++) {
	  index_l = nrec_*site_index(ic, jc, kc, nlocal);
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

void read_data(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index, nr;
  double phi;
  double revphi;


  if (input_binary_) {
    for (ic = 1; ic <= n[0]; ic++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (kc = 1; kc <= n[2]; kc++) {
	  index = site_index(ic, jc, kc, nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    fread(&phi, sizeof(double), 1, fp_data);
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
    for (ic = 1; ic <= n[0]; ic++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (kc = 1; kc <= n[2]; kc++) {
	  index = site_index(ic, jc, kc, nlocal);

	  for (nr = 0; nr < nrec_; nr++) {
	    fscanf(fp_data, "%le", data + nrec_*index + nr);
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

int le_displacement(int ic, int jc) {

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

void le_set_displacements() {

  int ic;
  int di;
  double dy;
  double duy;

  for (ic = 0; ic < ntotal[0]; ic++) {
    le_displacements_[ic] = 0.0;
    le_duy_[ic] = 0.0;
  }

  if (nplanes_ > 0) {

    di = ntotal[0] / nplanes_;
    dy = -(nplanes_/2.0)*le_displace_;
    duy = -(nplanes_/2.0)*le_speed_;

    /* Fist half block */
    for (ic = 1; ic <= di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
    }

    dy += le_displace_;
    duy += le_speed_;
    for (ic = di/2 + 1; ic <= ntotal[0] - di/2; ic++) {
      le_displacements_[ic-1] = dy;
      le_duy_[ic-1] = duy;
      if ( (ic - di/2) % di == 0 ) {
	dy += le_displace_;
	duy += le_speed_;
      }
    }

    /* Last half block */
    for (ic = ntotal[0] - di/2 + 1; ic <= ntotal[0]; ic++) {
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

void le_unroll(double * data) {

  int ic, jc, kc, n;
  int j0, j1, j2, j3, jdy;
  double * buffer;
  double dy, fr;
  double du[3];

  /* Allocate the temporary buffer */

  buffer = (double *) calloc(nrec_*ntargets[1]*ntargets[2], sizeof(double));
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
      j0 = 1 + (jc - jdy - 3 + 1000*ntotal[1]) % ntotal[1];
      j1 = 1 + j0 % ntotal[1];
      j2 = 1 + j1 % ntotal[1];
      j3 = 1 + j2 % ntotal[1];

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

int write_vtk_header(FILE * fp, int nrec, int ndim[3], const char * descript,
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
 *  write_qab_vtk
 *
 *  For Q_ab in vtk format, three separate files are required:
 *    scalar order parameter  (scalar)
 *    director                (vector)
 *    biaxial order parameter (scalar)
 *
 *****************************************************************************/

int write_qab_vtk(int ntime, int ntargets[3], double * datasection) {

  char io_data[FILENAME_MAX];
  FILE * fp_data = NULL;

  assert(datasection);
  assert(nrec_ == 5);

  /* Scalar order */
  if (output_lcs_) {
    sprintf(io_data, "lcq-%8.8d.vtk", ntime);
    fp_data = fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed scalar order with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntargets, "Q_ab_scalar_order", VTK_SCALARS);
    if (output_cmf_ == 0) write_data(fp_data, ntargets, 0, 1, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntargets, 0, 1, datasection);
    fclose(fp_data);
  }

  /* Director */
  fp_data = NULL;

  if (output_lcd_) {
    sprintf(io_data, "lcd-%8.8d.vtk", ntime);
    fp_data = fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed director with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntargets, "Q_ab_director", VTK_VECTORS);
    if (output_cmf_ == 0) write_data(fp_data, ntargets, 1, 3, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntargets, 1, 3, datasection);
    fclose(fp_data);
  }

  /* Biaxial order */
  fp_data = NULL;

  if (output_lcx_) {
    sprintf(io_data, "lcb-%8.8d.vtk", ntime);
    fp_data = fopen(io_data, "w");

    if (fp_data == NULL) {
      printf("fopen(%s) failed\n", io_data);
      exit(-1);
    }

    printf("Writing computed biaxial order with vtk: %s\n", io_data);

    write_vtk_header(fp_data, 1, ntargets, "Q_ab_biaxial_order", VTK_SCALARS);
    if (output_cmf_ == 0) write_data(fp_data, ntargets, 4, 1, datasection);
    if (output_cmf_ == 1) write_data_cmf(fp_data, ntargets, 4, 1, datasection);
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

    qs[0] = eigenvalue[0];
    qs[1] = eigenvector[0][0];
    qs[2] = eigenvector[1][0];
    qs[3] = eigenvector[2][0];

    s = eigenvalue[0];
    t = eigenvalue[1];

    q2 = s*s + t*t + (s + t)*(s + t);
    q3 = 3.0*s*t*(s + t);
    qs[4] = sqrt(1 - 6.0*q3*q3 / (q2*q2*q2));
  }

  return ifail;
}
