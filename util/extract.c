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
 *  Compile with $(CC) extract.c -lm
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nlocal[3];
int rank, cx, cy, cz, nlx, nly, nlz, nox, noy, noz;

int ntotal[3];
int ntargets[3]; /* Output target */
int pe_[3];
int nplanes_ = 0;
int nio_;
int nrec_ = 1;
int input_isbigendian_ = -1;   /* May need to deal with endianness */
int reverse_byte_order_ = 0;   /* Switch to deal with different endianess of input */
int input_binary_ = 1;         /* Switch for format of input */
int output_binary_ = 0;        /* Switch for format of final output */
int is_velocity_ = 0;          /* Switch to identify velocity field */

int le_t0_ = 0;                /* LE offset start time (time steps) */ 

int output_cmf_ = 0;           /* flag for output in column-major format */

double le_speed_ = 0.0;
double le_displace_ = 0.0;
double * le_displacements_;
double * le_duy_;

char stub_[FILENAME_MAX];

void read_meta_data_file(const char *);
int  read_data_file_name(const char *);

int site_index(int, int, int, const int *);
void read_data(FILE *, int *, double *);
void write_data(FILE *, int *, double *);
void write_data_cmf(FILE *, int *, double *);
int copy_data(double *, double *);
int le_displacement(int, int);
void le_set_displacements(void);
void le_unroll(double *);
double reverse_byte_order_double(char *);

int main(int argc, char ** argv) {

  int ntime;
  int i, n, p, pe_per_io;
  int ifail;

  double * datalocal;
  double * datasection;
  char io_metadata[FILENAME_MAX];
  char io_data[FILENAME_MAX];
  char line[FILENAME_MAX];

  FILE * fp_metadata;
  FILE * fp_data;

  /* Check the command line, then parse the meta data information,
   * and sort out the data file name  */

  if (argc != 3) {
    printf("Usage %s meta-file data-file\n", argv[0]);
    exit(-1);
  }

  read_meta_data_file(argv[1]);
  ntime = read_data_file_name(argv[2]);

  /* Work out parallel decompsoition that was used and the number of
   * processors per I/O group. */

  for (i = 0; i < 3; i++) {
    nlocal[i] = ntotal[i]/pe_[i];
  }

  pe_per_io = pe_[0]*pe_[1]*pe_[2]/nio_;


  /* Allocate storage */

  n = nrec_*nlocal[0]*nlocal[1]*nlocal[2];
  datalocal = (double *) calloc(n, sizeof(double));
  if (datalocal == NULL) printf("calloc(datalocal) failed\n");
    
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

  /* Unroll the data if Lees Edwards planes are present */

  if (nplanes_ > 0) {
    printf("Unrolling LE planes from centre (displacement %f)\n",
	   le_displace_);
    le_unroll(datasection);
  }

  /* Write a single file with the final section */

  sprintf(io_data, "%s-%8.8d", stub_, ntime);
  fp_data = fopen(io_data, "w+b");
  if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

  printf("\nWriting result to %s\n", io_data);

  if(output_cmf_ == 0) write_data(fp_data, ntargets, datasection);
  if(output_cmf_ == 1) write_data_cmf(fp_data, ntargets, datasection);

  fclose(fp_data);
  free(datalocal);
  free(le_displacements_);
  
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
  assert((nrbyte % 8) == 0);
  nrec_ = nrbyte/8;

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
 *  Write contiguous block of (float) data.
 *
 ****************************************************************************/

void write_data(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index, nr;

  index = 0;

  if (output_binary_) {
    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {
	  for (nr = 0; nr < nrec_; nr++) {
	    fwrite(data + index, sizeof(double), 1, fp_data);
	    index++;
	  }
	}
      }
    }
  }
  else {
    for (ic = 0; ic < n[0]; ic++) {
      for (jc = 0; jc < n[1]; jc++) {
	for (kc = 0; kc < n[2]; kc++) {
	  for (nr = 0; nr < nrec_ - 1; nr++) {
	    fprintf(fp_data, "%13.6e ", *(data + index));
	    index++;
	  }
	  fprintf(fp_data, "%13.6e\n", *(data + index));
	  index++;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  write_data_cmf
 *
 *  Loop around the sites in 'reverse' order (column major format).
 *
 *****************************************************************************/

void write_data_cmf(FILE * fp_data, int n[3], double * data) {

  int ic, jc, kc, index;
  int nr;

  if (output_binary_) {
    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {
	  index = site_index(ic, jc, kc, n);
	  for (nr = 0; nr < nrec_; nr++) {
	    fwrite(data + nrec_*index + nr, sizeof(double), 1, fp_data);
	  }
	}
      }
    }
  }
  else {
    for (kc = 1; kc <= n[2]; kc++) {
      for (jc = 1; jc <= n[1]; jc++) {
	for (ic = 1; ic <= n[0]; ic++) {
	  index = site_index(ic, jc, kc, n);
	  for (nr = 0; nr < nrec_ - 1; nr++) {
	    fprintf(fp_data, "%13.6e ", *(data + nrec_*index + nr));
	  }
	  fprintf(fp_data, "%13.6e\n", *(data + nrec_*index + nr));
	}
      }
    }
  }

  return;
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

  buffer = (double *) malloc(nrec_*ntargets[1]*ntargets[2]*sizeof(double));
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

/****************************************************************************
 *
 *  reverse_byte_order_double
 *
 *  Reverse the bytes in the char argument to make a double.
 *
 *****************************************************************************/

double reverse_byte_order_double(char * c) {

  double result;
  char * p = (char *) &result;
  int b;

  for (b = 0; b < sizeof(double); b++) {
    p[b] = c[sizeof(double) - (b + 1)];
  }

  return result;
}
