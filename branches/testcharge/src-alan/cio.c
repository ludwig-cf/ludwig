/*****************************************************************************
 *
 *  cio.c
 *
 *  Colloid parallel I/O driver.
 *
 *  $Id$
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
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group */
  int rank;                      /* Rank of PE in IO group */
  int single_file_read;          /* 'serial' input flag */
  int coords[3];                 /* Cartesian position of this group */

  MPI_Comm comm;                 /* Communicator */
  MPI_Comm xcomm;                /* Communicator between groups */

  void (* write_header_function)(FILE * fp, int nfile);
  void (* write_list_function)(FILE *, int, int, int);
  int  (* read_header_function)(FILE * fp);
  void (* read_list_function)(FILE * fp, int ndata);

} io_struct;

static io_struct cio_;

static void colloid_io_read_list_ascii(FILE * fp, int ndata);
static void colloid_io_read_list_binary(FILE * fp, int ndata);
static int  colloid_io_read_header_ascii(FILE *);
static int  colloid_io_read_header_binary(FILE *);
static void colloid_io_write_header_ascii(FILE *, int);
static void colloid_io_write_header_binary(FILE *, int);
static void colloid_io_write_list_ascii(FILE *, int, int, int);
static void colloid_io_write_list_binary(FILE *, int, int, int);

static int  colloid_io_count_colloids(void);
static void colloid_io_filename(char * filename, const char * stub);
static void colloid_io_check_read(int ngroup);

/*****************************************************************************
 *
 *  colloid_io_init
 *
 *  This split is the same as that which occurs in io_harness, which
 *  means the spatial decomposition is the same as that of the
 *  lattice quantities.
 *
 *  The io_grid[3] input is that requested; we return what actually
 *  gets allocated, given the constraints.
 *
 *****************************************************************************/

void colloid_io_init(int io_grid[3]) {

  int ia;
  int ncart;

  cio_.n_io = 1;

  for (ia = 0; ia < 3; ia++) {
    ncart = cart_size(ia);

    if (io_grid[ia] > ncart) io_grid[ia] = ncart;

    if (ncart % io_grid[ia] != 0) {
      fatal("Bad colloid io grid (dim %d = %d)\n", ia, io_grid[ia]);
    }

    cio_.n_io *= io_grid[ia];
    cio_.coords[ia] = io_grid[ia]*cart_coords(ia)/ncart;
  }

  cio_.index = cio_.coords[X] + cio_.coords[Y]*io_grid[X]
    + cio_.coords[Z]*io_grid[X]*io_grid[Y];

  MPI_Comm_split(cart_comm(), cio_.index, cart_rank(), &cio_.comm);
  MPI_Comm_rank(cio_.comm, &cio_.rank);
  MPI_Comm_size(cio_.comm, &cio_.size);

  cio_.single_file_read = 0;

  /* 'Cross' communicator between same rank in different groups. */

  MPI_Comm_split(cart_comm(), cio_.rank, cart_rank(), &cio_.xcomm);

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
 *  add up across the group. The root process of the group needs
 *  to know the total.
 *
 *****************************************************************************/

int colloid_io_count_colloids() {

  int nlocal;
  int ngroup;

  nlocal = colloid_nlocal();

  MPI_Reduce(&nlocal, &ngroup, 1, MPI_INT, MPI_SUM, 0, cio_.comm);

  return ngroup;
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
  int       ngroup;

  const int tag = 9572;

  MPI_Status status;

  if (colloid_ntotal() == 0) return;

  assert(cio_.write_header_function);
  assert(cio_.write_list_function);

  /* Set the filename */

  colloid_io_filename(filename_io, filename);

  info("\n");
  info("colloid_io_write:\n");
  info("writing colloid information to %s etc\n", filename_io);

  /* Make sure everyone has their current number of particles
   * up-to-date */

  ngroup = colloid_io_count_colloids();

  if (cio_.rank == 0) {
    /* Open the file, and write the header, followed by own colloids.
     * When this is done, pass the token to the next processs in the
     * group. */

    fp_state = fopen(filename_io, "w+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);

    cio_.write_header_function(fp_state, ngroup);
  }
  else {
    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can go ahead and append our own
     * colloids to the file. */

    MPI_Recv(&token, 1, MPI_INT, cio_.rank - 1, tag, cio_.comm, &status);
    fp_state = fopen(filename_io, "a+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }

  /* Write the local colloid state. */

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {
	cio_.write_list_function(fp_state, ic, jc, kc);
      }
    }
  }

  if (ferror(fp_state)) {
    perror("perror: ");
    fatal("Error on writing file %s\n", filename_io);
  }

  fclose(fp_state);

  /* This process has finished, so we can pass the token to the
   * next process in the group. If this is the last process in the
   * group, we've completed the file. */

  if (cio_.rank < cio_.size - 1) {
    /* Send the token */
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
 *  The read is a free-for-all in which all processes in the group
 *  read the entire file.
 *
 *****************************************************************************/

void colloid_io_read(const char * filename) {

  int    ngroup;
  char   filename_io[FILENAME_MAX];
  FILE * fp_state;

  assert(cio_.read_header_function);
  assert(cio_.read_list_function);

  /* Set the filename from the stub and the extension */

  colloid_io_filename(filename_io, filename);

  if (cio_.single_file_read) {
    /* All groups read for single 'serial' file */
    sprintf(filename_io, "%s.%3.3d-%3.3d", filename, 1, 1);
    info("colloid_io_read: reading from single file %s\n", filename_io);
  }
  else {
    info("colloid_io_read: reading from %s etc\n", filename_io);
  }

  /* Open the file and read the information */

  fp_state = fopen(filename_io, "r");
  if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);

  ngroup = cio_.read_header_function(fp_state);
  cio_.read_list_function(fp_state, ngroup);

  if (ferror(fp_state)) {
    perror("perror: ");
    fatal("Error on reading %s\n", filename_io);
  }

  fclose(fp_state);

  colloid_io_check_read(ngroup);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_ascii
 *
 *  Write colloid information to file in human-readable form.
 *
 *****************************************************************************/

static void colloid_io_write_header_ascii(FILE * fp, int ngroup) {

  fprintf(fp, "%22d\n", ngroup);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_ascii
 *
 *  Reads the number of particles in the file.
 *
 *****************************************************************************/

static int colloid_io_read_header_ascii(FILE * fp) {

  int nfile;

  fscanf(fp, "%22d\n",  &nfile);

  return nfile;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_binary
 *
 *  Write the colloid header information to file.
 *
 *****************************************************************************/

static void colloid_io_write_header_binary(FILE * fp, int ngroup) {

  fwrite(&ngroup, sizeof(int), 1, fp);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_binary
 *
 *  The read equivalent of the above.
 *
 *****************************************************************************/

static int colloid_io_read_header_binary(FILE * fp) {

  int nfile;

  fread(&nfile, sizeof(int), 1, fp);

  return nfile;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_ascii
 *
 *****************************************************************************/

static void colloid_io_write_list_ascii(FILE * fp, int ic, int jc, int kc) {

  colloid_t * p_colloid;

  p_colloid = colloids_cell_list(ic, jc, kc);

  while (p_colloid) {
    colloid_state_write_ascii(p_colloid->s, fp);
    p_colloid = p_colloid->next;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_ascii
 *
 *****************************************************************************/

static void colloid_io_read_list_ascii(FILE * fp, int ndata) {

  int nread;
  int nlocal = 0;

  colloid_state_t s;
  colloid_t * p_colloid;

  for (nread = 0; nread < ndata; nread++) {
    colloid_state_read_ascii(&s, fp);
    p_colloid = colloid_add_local(s.index, s.r);
    if (p_colloid) {
      p_colloid->s = s;
      nlocal++;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_binary
 *
 *****************************************************************************/

static void colloid_io_write_list_binary(FILE * fp, int ic, int jc, int kc) {

  colloid_t * p_colloid;

  p_colloid = colloids_cell_list(ic, jc, kc);

  while (p_colloid) {
    colloid_state_write_binary(p_colloid->s, fp);
    p_colloid = p_colloid->next;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_binary
 *
 *****************************************************************************/

static void colloid_io_read_list_binary(FILE * fp, int ndata) {

  int nread;

  colloid_state_t s;
  colloid_t * p_colloid;

  for (nread = 0; nread < ndata; nread++) {
    colloid_state_read_binary(&s, fp);
    p_colloid = colloid_add_local(s.index, s.r);
    if (p_colloid)  p_colloid->s = s;
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
 *  colloid_io_format_input_serial_set
 *
 *****************************************************************************/

void colloid_io_format_input_serial_set(void) {

  cio_.single_file_read = 1;

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
 *  colloid_io_filename
 *
 *  Add the extension to the supplied stub.
 *
 *****************************************************************************/

static void colloid_io_filename(char * filename, const char * stub) {

  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* Check stub not too long */

  if (cio_.index >= 1000) fatal("Format botch for cio stub %s\n", stub);

  sprintf(filename, "%s.%3.3d-%3.3d", stub, cio_.n_io, cio_.index + 1);

  return;
}

/*****************************************************************************
 *
 *  colloid_io_check_read
 *
 *  Check the number of colloids in the list is consistent with that
 *  in the file (ngroup), and set the total.
 *
 *  If we haven't lost any particles, we can set the global total and
 *  proceed.
 *
 *****************************************************************************/

static void colloid_io_check_read(int ngroup) {

  int nlocal;
  int ntotal;

  nlocal = colloid_nlocal();

  MPI_Reduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, 0, cio_.comm);

  if (cio_.single_file_read) {
    /* Only the global total can be compared (ngroup is ntotal). */
    nlocal = ntotal;
    MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, cio_.xcomm);
  }

  if (cio_.rank == 0) {
    if (ntotal != ngroup) {
      verbose("Colloid I/O group %d\n", cio_.index);
      verbose("Colloids in file: %d Got %d\n", ngroup, ntotal);
      fatal("Total number of colloids not consistent with file\n");
    }
  }

  colloids_ntotal_set();
  info("Read a total of %d colloids from file\n", colloid_ntotal());


  return;
}
