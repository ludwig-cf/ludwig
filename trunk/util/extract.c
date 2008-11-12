/****************************************************************************
 *
 *  extract.c
 *
 *  This program deals with the consolidation of parallel I/O
 *  (in serial).
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
 *  $Id: extract.c,v 1.3 2008-11-12 15:12:39 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Grouand and
 *  Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh (2008)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
int input_isbigendian_ = -1;
double le_speed_ = 0.0;
int le_displace_ = 0;
int * le_displacements_;
int output_binary_ = 0;

char stub_[FILENAME_MAX];

void read_meta_data_file(const char *);
int  read_data_file_name(const char *);

int site_index(int, int, int, const int *);
void read_data(FILE *, int *, double *);
void write_data(FILE *, int *, double *);
int copy_data(double *, double *);
int le_displacement(int, int);
void le_set_displacements(void);
void le_unroll(double *);

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
  le_displacements_ = (int *) malloc(ntotal[0]*sizeof(int));
  if (le_displacements_ == NULL) printf("malloc(le_displacements_)\n");
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

    sprintf(io_data, "%s-%6.6d.%3.3d-%3.3d", stub_, ntime, nio_, n);
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

  /* Write a single file with the final section */

  sprintf(io_data, "%s-%6.6d", stub_, ntime);
  fp_data = fopen(io_data, "w+b");
  if (fp_data == NULL) printf("fopen(%s) failed\n", io_data);

  printf("\nWriting result to %s\n", io_data);

  write_data(fp_data, ntargets, datasection);

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

  for (ic = 1; ic <= n[0]; ic++) {
    for (jc = 1; jc <= n[1]; jc++) {
      for (kc = 1; kc <= n[2]; kc++) {
	index = site_index(ic, jc, kc, nlocal);

	for (nr = 0; nr < nrec_; nr++) {
	  fread(&phi, sizeof(double), 1, fp_data);
	  *(data + nrec_*index + nr) = phi;
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

  int ic, dy;
  int di;

  for (ic = 0; ic < ntotal[0]; ic++) {
    le_displacements_[ic] = 0.0;
  }

  if (nplanes_ > 0) {

    di = ntotal[0] / nplanes_;
    dy = -(nplanes_/2)*le_displace_;

    /* Fist half block */
    for (ic = 1; ic <= di/2; ic++) {
      le_displacements_[ic-1] = dy;
    }

    dy += le_displace_;
    for (ic = di/2 + 1; ic <= ntotal[0] - di/2; ic++) {
      le_displacements_[ic-1] = dy;
      if ( (ic - di/2) % di == 0 ) dy += le_displace_;
    }

    /* Last half block */
    for (ic = ntotal[0] - di/2 + 1; ic <= ntotal[0]; ic++) {
      le_displacements_[ic-1] = dy;
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
 *****************************************************************************/

void le_unroll(double * data) {

  int ic, jc, kc;
  int j1, j2, jdy;
  double * buffer;
  double dy, fr;

  assert(nrec_ <= 1);

  /* Allocate the temporary buffer */
    
  buffer = (double *) malloc(ntargets[1]*ntargets[2]*sizeof(double));
  if (buffer == NULL) {
    printf("malloc(buffer) failed\n");
    exit(-1);
  }

  for (ic = 1; ic <= ntargets[0]; ic++) {
    dy = le_displacements_[ic-1];
    jdy = floor(dy);
    fr = dy - jdy;

    for (jc = 1; jc <= ntargets[1]; jc++) {
      j1 = 1 + (jc - jdy - 2 + 100*ntotal[1]) % ntotal[1];
      j2 = 1 + j1 % ntotal[1];

      for (kc = 1; kc <= ntargets[2]; kc++) {
	buffer[site_index(1,jc,kc,ntargets)] = 
	  fr*data[site_index(ic,j1,kc,ntargets)] +
	  (1.0 - fr)*data[site_index(ic,j2,kc,ntargets)];
      }
    }
    /* Put the whole buffer plane back in place */

    for (jc = 1; jc <= ntargets[1]; jc++) {
      for (kc = 1; kc <= ntargets[2]; kc++) {
	data[site_index(ic,jc,kc,ntargets)] =
	  buffer[site_index(1,jc,kc,ntargets)];
      }
    }

  }

  free(buffer);

  return;
}
