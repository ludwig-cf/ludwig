/*****************************************************************************
 *
 *  cio.c
 *
 *  Colloid I/O, serial and parallel.
 *
 *  $Id: cio.c,v 1.7.16.4 2010-05-19 19:16:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "cio.h"

typedef struct {
  int root;                      /* Root PE of current I/O group */
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group */
  int rank;                      /* Rank of PE in IO group */

  MPI_Comm comm;                 /* Communicator */

  void (* write_header_function)(FILE *);
  int  (* write_list_function)(FILE *, int, int, int);
  void (* read_header_function)(FILE *);
  void (* read_list_function)(FILE *);

} io_struct;

static io_struct cio_;
static int nlocal_;                       /* Local number of colloids. */
static int ntotal_;

static void colloid_io_read_list_ascii(FILE *);
static void colloid_io_read_list_ascii_serial(FILE *);
static void colloid_io_read_list_binary(FILE *);
static void colloid_io_read_header_ascii(FILE *);
static void colloid_io_read_header_binary(FILE *);
static void colloid_io_write_header_ascii(FILE *);
static void colloid_io_write_header_binary(FILE *);
static int  colloid_io_write_list_ascii(FILE *, int, int, int);
static int  colloid_io_write_list_binary(FILE *, int, int, int);

static void colloid_io_count_colloids(void);
static void cio_filename(char * filename, const char * stub);

/*****************************************************************************
 *
 *  colloid_io_init
 *
 *****************************************************************************/

void colloid_io_init(void) {

  cio_.n_io = 1; /* Always 1 at moment */

  if (cio_.n_io > pe_size()) cio_.n_io = pe_size();
  cio_.size = pe_size() / cio_.n_io;

  if ((cart_rank() % cio_.size) == 0) {
    cio_.root = 1;
  }
  else {
    cio_.root = 0;
  }

  cio_.index = cart_rank()/cio_.size;

  /* Create communicator for each IO group, and get rank within IO group */
  MPI_Comm_split(cart_comm(), cio_.index, cart_rank(), &cio_.comm);
  MPI_Comm_rank(cio_.comm, &cio_.rank);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_finish
 *
 *****************************************************************************/

void colloid_io_finish(void) {

  MPI_Comm_free(&cio_.comm);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_count_colloids
 *
 *  Count the local number of (physical, not halo) colloids and
 *  add up across proceses.
 *
 *****************************************************************************/

void colloid_io_count_colloids() {

  nlocal_ = colloid_nlocal();

  MPI_Allreduce(&nlocal_, &ntotal_, 1, MPI_INT, MPI_SUM, cart_comm());

  assert(ntotal_ == colloid_ntotal());

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write
 *
 *  Write information on the colloids in the local domain proper (not
 *  including the halos) to the specified file.
 *
 *  In parallel, one file per io group is used. Processes within a
 *  group write to the same file on reciept of the token.
 *
 *****************************************************************************/

void colloid_io_write(const char * filename) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  int       token = 0;
  int       ic, jc, kc;

  const int tag = 9572;

  MPI_Status status;

  if (colloid_ntotal() == 0) return;

  assert(cio_.write_header_function);
  assert(cio_.write_list_function);

  /* Set the filename */

  cio_filename(filename_io, filename);

  info("colloid_io_write:\n");
  info("writing colloid information to %s etc\n", filename_io);

  /* Make sure everyone has their current number of particles
   * up-to-date */

  colloid_io_count_colloids();

  if (cio_.root) {
    /* Open the file, and write the header, followed by own colloids.
     * When this is done, pass the token to the next processs in the
     * group. */

    fp_state = fopen(filename_io, "w+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }
  else {
    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can go ahead and append our own
     * colloids to the file. */

    MPI_Recv(&token, 1, MPI_INT, cio_.rank - 1, tag, cio_.comm, &status);
    fp_state = fopen(filename_io, "a+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }

  /* Write the local colloid state, consisting of header plus data. */

  cio_.write_header_function(fp_state);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	token += cio_.write_list_function(fp_state, ic, jc, kc);
      }
    }
  }

  fclose(fp_state);

  /* This process has finished, so we can pass the token to the
   * next process in the group. If this is the last process in the
   * group, we've completed the file. */

  if (cio_.rank < cio_.size - 1) {
    /* Send the token, which happens to be the accumulated number of
     * particles. */

    MPI_Ssend(&token, 1, MPI_INT, cio_.rank + 1, tag, cio_.comm);
  }

  return;
}


/*****************************************************************************
 *
 *  colloid_io_read
 *
 *  This is the driver routine to read colloid information from file.
 *
 *****************************************************************************/

void colloid_io_read(const char * filename) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  long int  token = 0;
  const int tag = 9573;

  MPI_Status status;

  assert(cio_.read_header_function);
  assert(cio_.read_list_function);

  /* Set the filename from the stub and the extension */

  info("colloid_io_read:\n");
  cio_filename(filename_io, filename);

  if (cio_.root) {
    /* Open the file and read the header information */
    fp_state = fopen(filename_io, "r");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }
  else {
    /* Block until we receive the token which allows us to proceed.
     * Then, open the file and move to the appropriate position before
     * starting to read. */

    MPI_Recv(&token, 1, MPI_LONG, cio_.rank - 1, tag, cio_.comm, &status);

    fp_state = fopen(filename_io, "r");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
    rewind(fp_state);

    /* This is a fix for reading serial files in parallel */
    if (cio_.read_list_function != colloid_io_read_list_ascii_serial) {
      fseek(fp_state, token, SEEK_SET);
    }
  }

  /* Read the data */

  cio_.read_header_function(fp_state);
  cio_.read_list_function(fp_state);

  token = ftell(fp_state);
  fclose(fp_state);

  /* Pass on the token, which is the current offset in the file */

  if (cio_.rank < cio_.size - 1) {

    MPI_Ssend(&token, 1, MPI_LONG, cio_.rank + 1, tag, cio_.comm);

  }

  /* This is set here, as the total is not yet known. */
  set_N_colloid(ntotal_);
  info("Reading information for %d particles\n", ntotal_);

  return;
}


/*****************************************************************************
 *
 *  colloid_io_write_header_ascii
 *
 *  Write colloid information to file in human-readable form.
 *
 *****************************************************************************/

static void colloid_io_write_header_ascii(FILE * fp) {

  fprintf(fp, "I/O n_io:  %22d\n", cio_.n_io);
  fprintf(fp, "I/O index: %22d\n", cio_.index);
  fprintf(fp, "N_colloid: %22d\n", ntotal_);
  fprintf(fp, "nlocal:    %22d\n", nlocal_);

  return;
}


/*****************************************************************************
 *
 *  colloid_io_read_header_ascii
 *
 *  Everybody reads their own header, which should include the
 *  local number of particles.
 *
 *  Issues
 *    A large nuumber of these values are ignored at the moment,
 *    meaning that run time input must agree. There is an outstanding
 *    issue as to what to do if one wants to override run time input.
 *
 *****************************************************************************/

static void colloid_io_read_header_ascii(FILE * fp) {

  int    read_int;

  info("Colloid file header information has been filtered\n");

  fscanf(fp, "I/O n_io:  %22d\n",  &read_int);
  fscanf(fp, "I/O index: %22d\n",  &read_int);
  fscanf(fp, "N_colloid: %22d\n",  &ntotal_);
  fscanf(fp, "nlocal:    %22d\n",  &nlocal_);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_binary
 *
 *  Write the colloid header information to file.
 *
 *****************************************************************************/

static void colloid_io_write_header_binary(FILE * fp) {

  fwrite(&(cio_.n_io),   sizeof(int),     1, fp);
  fwrite(&(cio_.index),  sizeof(int),     1, fp);
  fwrite(&ntotal_,         sizeof(int),     1, fp);
  fwrite(&nlocal_,         sizeof(int),     1, fp);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_binary
 *
 *  The read equivalent of the above.
 *
 *****************************************************************************/

static void colloid_io_read_header_binary(FILE * fp) {

  int     read_int;

  fread(&(read_int), sizeof(int),     1, fp); /* n_io */
  fread(&(read_int), sizeof(int),     1, fp);
  fread(&ntotal_,    sizeof(int),     1, fp);
  fread(&nlocal_,    sizeof(int),     1, fp);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_ascii
 *
 *****************************************************************************/

static int colloid_io_write_list_ascii(FILE * fp, int ic, int jc, int kc) {

  Colloid * p_colloid;
  int       nwrite = 0;

  p_colloid = CELL_get_head_of_list(ic, jc, kc);

  while (p_colloid) {
    nwrite++;

    fprintf(fp, "%22.15e %22.15e %d\n", p_colloid->a0, p_colloid->ah,
	    p_colloid->index);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->r[X], p_colloid->r[Y],
	    p_colloid->r[Z]);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->v[X], p_colloid->v[Y],
	    p_colloid->v[Z]);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->omega[X],
	    p_colloid->omega[Y], p_colloid->omega[Z]);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->s[X], p_colloid->s[Y],
            p_colloid->s[Z]);
    fprintf(fp, "%22.15e %22.15e %22.15e\n", p_colloid->direction[X],
	    p_colloid->direction[Y], p_colloid->direction[Z]);
    fprintf(fp, "%22.15e\n", p_colloid->b1);
    fprintf(fp, "%22.15e\n", p_colloid->b2);
    fprintf(fp, "%22.15e\n", p_colloid->c_wetting);
    fprintf(fp, "%22.15e\n", p_colloid->h_wetting);
    fprintf(fp, "%22.15e\n", p_colloid->deltaphi);
 
    /* Next colloid */
    p_colloid = p_colloid->next;
  }

  return nwrite;
}


/*****************************************************************************
 *
 *  colloid_io_read_ascii
 *
 *  The colloid data from file an add a new particle to the local
 *  list.
 *
 *****************************************************************************/

static void colloid_io_read_list_ascii(FILE * fp) {

  int       nread;
  int       read_index;
  double    read_a0;
  double    read_ah;
  double    read_r[3];
  double    read_v[3];
  double    read_s;
  Colloid * p_colloid;

  for (nread = 0; nread < nlocal_; nread++) {

    fscanf(fp, "%22le %22le %22d\n",  &(read_a0), &(read_ah), &(read_index));
    fscanf(fp, "%22le %22le %22le\n", read_r, read_r + 1, read_r + 2);

    p_colloid = colloid_add(read_index, read_r);

    if (p_colloid == NULL) {
      /* This didn't go into the cell list */
      fatal("Colloid information doesn't tally with read position\n");
    }

    p_colloid->a0 = read_a0;
    p_colloid->ah = read_ah;

    fscanf(fp, "%22le %22le %22le\n", read_v, read_v + 1, read_v + 2);
    p_colloid->v[X] = read_v[X];
    p_colloid->v[Y] = read_v[Y];
    p_colloid->v[Z] = read_v[Z];

    fscanf(fp, "%22le %22le %22le\n", read_v, read_v + 1, read_v + 2);
    p_colloid->omega[X] = read_v[X];
    p_colloid->omega[Y] = read_v[Y];
    p_colloid->omega[Z] = read_v[Z];

    fscanf(fp, "%22le %22le %22le\n", read_v, read_v + 1, read_v + 2);
    p_colloid->s[X] = read_v[X];
    p_colloid->s[Y] = read_v[Y];
    p_colloid->s[Z] = read_v[Z];

    fscanf(fp, "%22le %22le %22le\n", read_v, read_v + 1, read_v + 2);
    p_colloid->direction[X] = read_v[X];
    p_colloid->direction[Y] = read_v[Y];
    p_colloid->direction[Z] = read_v[Z];

    fscanf(fp, "%22le\n", &read_s);
    p_colloid->b1 = read_s;
    fscanf(fp, "%22le\n", &read_s);
    p_colloid->b2 = read_s;

    fscanf(fp, "%22le\n", &read_s);
    p_colloid->c_wetting = read_s;
    fscanf(fp, "%22le\n", &read_s);
    p_colloid->h_wetting = read_s;

    fscanf(fp, "%22le\n", &read_s);
    p_colloid->deltaphi = read_s;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_ascii_serial
 *
 *  This is a solution to reading a serial file in parallel.
 *  Each process reads all the particles, and throws away
 *  those not in the local domain.
 *
 *****************************************************************************/

static void colloid_io_read_list_ascii_serial(FILE * fp) {

  int       nread;
  int       read_index;
  double    read_a0;
  double    read_ah;
  double    read_s[3];
  double    read_d[3];
  double    read_c, read_h;
  double    read_deltaphi;
  Colloid * p_colloid;

  for (nread = 0; nread < ntotal_; nread++) {

    fscanf(fp, "%22le %22le %22d\n",  &(read_a0), &(read_ah), &(read_index));
    fscanf(fp, "%22le %22le %22le\n", read_s, read_s+1, read_s+2);

    p_colloid = colloid_add_local(read_index, read_s);

    if (p_colloid) {
      fscanf(fp, "%22le %22le %22le\n", read_s, read_s+1, read_s+2);
      p_colloid->v[X] = read_s[X];
      p_colloid->v[Y] = read_s[Y];
      p_colloid->v[Z] = read_s[Z];

      fscanf(fp, "%22le %22le %22le\n", read_s, read_s+1, read_s+2);
      p_colloid->omega[X] = read_s[X];
      p_colloid->omega[Y] = read_s[Y];
      p_colloid->omega[Z] = read_s[Z];
    }

    if (p_colloid) {
      /* Read and store */
      fscanf(fp, "%22le %22le %22le\n", read_s, read_s+1, read_s+2);
      fscanf(fp, "%22le %22le %22le\n", read_d, read_d+1, read_d+2);
      fscanf(fp, "%22le\n",             &read_c);
      fscanf(fp, "%22le\n",             &read_h);
      fscanf(fp, "%22le\n",             &(read_deltaphi));
      p_colloid->deltaphi = read_deltaphi;
      p_colloid->s[X] = read_s[X];
      p_colloid->s[Y] = read_s[Y];
      p_colloid->s[Z] = read_s[Z];
      p_colloid->direction[X] = read_d[X];
      p_colloid->direction[Y] = read_d[Y];
      p_colloid->direction[Z] = read_d[Z];
      p_colloid->c_wetting = read_c;
      p_colloid->h_wetting = read_h;
    }
    else {
      /* Read anyway, and ignore TODO */
      fatal("Unhandled non-local colloid reads\n");
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_binary
 *
 *  Write out the colliod information (if any) for the specified list.
 *
 *****************************************************************************/

static int colloid_io_write_list_binary(FILE * fp, int ic, int jc, int kc) {

  Colloid * p_colloid;
  int       nwrite = 0;

  p_colloid = CELL_get_head_of_list(ic, jc, kc);

  while (p_colloid) {
    nwrite++;
    fwrite(&(p_colloid->a0),       sizeof(double),  1, fp);
    fwrite(&(p_colloid->ah),       sizeof(double),  1, fp);
    fwrite(&(p_colloid->index),    sizeof(int),     1, fp);
    fwrite(&(p_colloid->r),        sizeof(double), 3, fp);
    fwrite(&(p_colloid->v),        sizeof(double), 3, fp);
    fwrite(&(p_colloid->omega),    sizeof(double), 3, fp);
    fwrite(p_colloid->dr,          sizeof(double),  3, fp);
    fwrite(p_colloid->s,           sizeof(double),  3, fp);
    fwrite(p_colloid->direction,   sizeof(double),  3, fp);
    fwrite(&(p_colloid->c_wetting), sizeof(double),  1, fp);
    fwrite(&(p_colloid->h_wetting), sizeof(double),  1, fp);
    fwrite(&(p_colloid->deltaphi), sizeof(double),  1, fp);

    /* Next colloid */
    p_colloid = p_colloid->next;
  }

  return nwrite;
}


/*****************************************************************************
 *
 *  colloid_io_read_list_binary
 *
 *****************************************************************************/

static void colloid_io_read_list_binary(FILE * fp) {

  int       nread;
  int       read_index;
  double    read_a0;
  double    read_ah;
  double    read_dr[3];
  double    read_s[3];
  double    read_c, read_h;
  double    read_deltaphi;
  Colloid * p_colloid;

  for (nread = 0; nread < nlocal_; nread++) {

    fread(&read_a0,       sizeof(double),  1, fp);
    fread(&read_ah,       sizeof(double),  1, fp);
    fread(&read_index,    sizeof(int),     1, fp);
    fread(read_dr,        sizeof(double),  3, fp);
    fread(read_s,         sizeof(double),  3, fp);
    fread(&read_c,        sizeof(double),  1, fp);
    fread(&read_h,        sizeof(double),  1, fp);
    fread(&read_deltaphi, sizeof(double),  1, fp);

    fatal("Forgot this\n");
    /* Add colloid */
    p_colloid = NULL;

    if (p_colloid) {
      int i;
      p_colloid->deltaphi = read_deltaphi;
      for (i = 0; i < 3; i++) {
	p_colloid->dr[i] = read_dr[i];
	p_colloid->s[i] = read_s[i];
      }
      p_colloid->c_wetting = read_c;
      p_colloid->h_wetting = read_h;
    }
    else {
      /* This didn't go into the cell list */
      fatal("Colloid information doesn't tally with read position\n");
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_ascii_set
 *
 *****************************************************************************/

void colloid_io_format_input_ascii_set(void) {

  cio_.read_header_function = colloid_io_read_header_ascii;
  cio_.read_list_function = colloid_io_read_list_ascii;

  return;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_binary_set
 *
 *****************************************************************************/

void colloid_io_format_input_binary_set(void) {

  cio_.read_header_function = colloid_io_read_header_binary;
  cio_.read_list_function = colloid_io_read_list_binary;

  return;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_ascii_serial_set
 *
 *****************************************************************************/

void colloid_io_format_input_ascii_serial_set(void) {

  cio_.read_header_function = colloid_io_read_header_ascii;
  cio_.read_list_function = colloid_io_read_list_ascii;

  return;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_ascii_set
 *
 *****************************************************************************/

void colloid_io_format_output_ascii_set(void) {

  cio_.write_header_function = colloid_io_write_header_ascii;
  cio_.write_list_function = colloid_io_write_list_ascii;

  return;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_binary_set
 *
 *****************************************************************************/

void colloid_io_format_output_binary_set(void) {

  cio_.write_header_function = colloid_io_write_header_binary;
  cio_.write_list_function = colloid_io_write_list_binary;

  return;
}

/*****************************************************************************
 *
 *  cio_filename
 *
 *  Add the extension to the supplied stub.
 *
 *****************************************************************************/

static void cio_filename(char * filename, const char * stub) {

  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* Check stub not too long */

  if (cio_.index >= 1000) fatal("Format botch for cio stub %s\n", stub);

  sprintf(filename, "%s.%3.3d-%3.3d", stub, cio_.n_io, cio_.index + 1);

  return;
}
