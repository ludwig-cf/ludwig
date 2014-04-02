/*****************************************************************************
 *
 *  colloid_io.c
 *
 *  Colloid parallel I/O driver.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2013 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "colloid_io.h"

struct colloid_io_s {
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group */
  int rank;                      /* Rank of PE in IO group */
  int single_file_read;          /* 'serial' input flag */
  int coords[3];                 /* Cartesian position of this group */

  MPI_Comm comm;                 /* Communicator */
  MPI_Comm xcomm;                /* Communicator between groups */

  int (* f_header_write) (FILE * fp, int nfile);
  int (* f_header_read)  (FILE * fp, int * nfile);
  int (* f_list_write)   (colloid_io_t * cio, int, int, int, FILE * fp);
  int (* f_list_read)    (colloid_io_t * cio, int ndata, FILE * fp);

  colloids_info_t * info;        /* Keep a reference to this */
};

static int colloid_io_read_header_binary(FILE * fp, int * nfile);
static int colloid_io_read_header_ascii(FILE * fp, int * nfile);
static int colloid_io_write_header_binary(FILE *, int nfile);
static int colloid_io_write_header_ascii(FILE *, int nfile);

static int colloid_io_read_list_ascii(colloid_io_t * cio, int nc, FILE * fp);
static int colloid_io_read_list_binary(colloid_io_t * cio, int nc, FILE * fp);
static int colloid_io_write_list_ascii(colloid_io_t * cio,
				       int ic , int jc, int kc, FILE * fp);
static int colloid_io_write_list_binary(colloid_io_t * cio,
					int ic , int jc, int kc, FILE * fp);

static int colloid_io_count_colloids(colloid_io_t * cio, int * ngroup);
static int colloid_io_filename(colloid_io_t * cio, char * filename,
			       const char * stub);
static int colloid_io_check_read(colloid_io_t * cio, int ngroup);

/*****************************************************************************
 *
 *  colloid_io_create
 *
 *  This split is the same as that which occurs in io_harness, which
 *  means the spatial decomposition is the same as that of the
 *  lattice quantities.
 *
 *  The io_grid[3] input is that requested; we return what actually
 *  gets allocated, given the constraints.
 *
 *****************************************************************************/

int colloid_io_create(int io_grid[3], colloids_info_t * info,
		      colloid_io_t ** pcio) {
  int ia;
  int ncart;
  colloid_io_t * obj = NULL;

  assert(info);
  assert(pcio);

  obj = calloc(1, sizeof(colloid_io_t));
  if (obj == NULL) fatal("calloc(colloid_io_t) failed\n");

  obj->n_io = 1;

  for (ia = 0; ia < 3; ia++) {
    ncart = cart_size(ia);

    if (io_grid[ia] > ncart) io_grid[ia] = ncart;

    if (ncart % io_grid[ia] != 0) {
      fatal("Bad colloid io grid (dim %d = %d)\n", ia, io_grid[ia]);
    }

    obj->n_io *= io_grid[ia];
    obj->coords[ia] = io_grid[ia]*cart_coords(ia)/ncart;
  }

  obj->index = obj->coords[X] + obj->coords[Y]*io_grid[X]
    + obj->coords[Z]*io_grid[X]*io_grid[Y];

  MPI_Comm_split(cart_comm(), obj->index, cart_rank(), &obj->comm);
  MPI_Comm_rank(obj->comm, &obj->rank);
  MPI_Comm_size(obj->comm, &obj->size);

  /* 'Cross' communicator between same rank in different groups. */

  MPI_Comm_split(cart_comm(), obj->rank, cart_rank(), &obj->xcomm);

  obj->info = info;
  *pcio = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_finish
 *
 *****************************************************************************/

void colloid_io_finish(colloid_io_t * cio) {

  assert(cio);

  MPI_Comm_free(&cio->xcomm);
  MPI_Comm_free(&cio->comm);

  free(cio);

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

int colloid_io_count_colloids(colloid_io_t * cio, int * ngroup) {

  int nlocal;

  assert(cio);
  assert(ngroup);

  colloids_info_nlocal(cio->info, &nlocal);
  MPI_Reduce(&nlocal, ngroup, 1, MPI_INT, MPI_SUM, 0, cio->comm);

  return 0;
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

int colloid_io_write(colloid_io_t * cio, const char * filename) {

  int ntotal;
  int ngroup;
  int token = 0;
  int ic, jc, kc;
  int ncell[3];
  const int tag = 9572;

  char filename_io[FILENAME_MAX];
  FILE * fp_state = NULL;
  MPI_Status status;

  assert(cio);

  colloids_info_ntotal(cio->info, &ntotal);
  if (ntotal == 0) return 0;

  assert(cio->f_header_write);
  assert(cio->f_list_write);
  /* Set the filename */

  colloid_io_filename(cio, filename_io, filename);

  info("\n");
  info("colloid_io_write:\n");
  info("writing colloid information to %s etc\n", filename_io);

  /* Make sure everyone has their current number of particles
   * up-to-date */

  colloid_io_count_colloids(cio, &ngroup);

  if (cio->rank == 0) {

    /* Open the file, and write the header, followed by own colloids.
     * When this is done, pass the token to the next processs in the
     * group. */

    fp_state = fopen(filename_io, "w+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);

    cio->f_header_write(fp_state, ngroup);
  }
  else {
    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can go ahead and append our own
     * colloids to the file. */

    MPI_Recv(&token, 1, MPI_INT, cio->rank - 1, tag, cio->comm, &status);
    fp_state = fopen(filename_io, "a+b");
    if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  }

  /* Write the local colloid state. */

  colloids_info_ncell(cio->info, ncell);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {
	cio->f_list_write(cio, ic, jc, kc, fp_state);
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

  if (cio->rank < cio->size - 1) {
    /* Send the token */
    MPI_Ssend(&token, 1, MPI_INT, cio->rank + 1, tag, cio->comm);
  }

  return 0;
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

int colloid_io_read(colloid_io_t * cio, const char * filename) {

  int    ngroup;
  char   filename_io[FILENAME_MAX];
  FILE * fp_state;

  assert(cio->f_header_read);
  assert(cio->f_list_read);

  /* Set the filename from the stub and the extension */

  colloid_io_filename(cio, filename_io, filename);

  if (cio->single_file_read) {
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

  cio->f_header_read(fp_state, &ngroup);
  cio->f_list_read(cio, ngroup, fp_state);

  if (ferror(fp_state)) {
    perror("perror: ");
    fatal("Error on reading %s\n", filename_io);
  }

  fclose(fp_state);

  colloid_io_check_read(cio, ngroup);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_ascii
 *
 *  Write colloid information to file in human-readable form.
 *
 *****************************************************************************/

static int colloid_io_write_header_ascii(FILE * fp, int ngroup) {

  fprintf(fp, "%22d\n", ngroup);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_ascii
 *
 *  Reads the number of particles in the file.
 *
 *****************************************************************************/

static int colloid_io_read_header_ascii(FILE * fp, int * nfile) {

  assert(nfile);

  fscanf(fp, "%22d\n",  nfile);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_header_binary
 *
 *  Write the colloid header information to file.
 *
 *****************************************************************************/

static int colloid_io_write_header_binary(FILE * fp, int ngroup) {

  fwrite(&ngroup, sizeof(int), 1, fp);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_header_binary
 *
 *  The read equivalent of the above.
 *
 *****************************************************************************/

static int colloid_io_read_header_binary(FILE * fp, int * nfile) {

  assert(fp);
  assert(nfile);

  fread(nfile, sizeof(int), 1, fp);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_ascii
 *
 *****************************************************************************/

static int colloid_io_write_list_ascii(colloid_io_t * cio,
				       int ic, int jc, int kc, FILE * fp) {
  int ifail = 0;
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  colloids_info_cell_list_head(cio->info, ic, jc, kc, &pc);

  while (pc) {
    ifail += colloid_state_write_ascii(pc->s, fp);
    pc = pc->next;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_ascii
 *
 *****************************************************************************/

static int colloid_io_read_list_ascii(colloid_io_t * cio, int ndata,
				      FILE * fp) {
  int nread;
  int nlocal = 0;

  colloid_state_t s;
  colloid_t * p_colloid;

  assert(cio);
  assert(fp);

  for (nread = 0; nread < ndata; nread++) {
    colloid_state_read_ascii(&s, fp);
    colloids_info_add_local(cio->info, s.index, s.r, &p_colloid);
    if (p_colloid) {
      p_colloid->s = s;
      nlocal++;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_write_list_binary
 *
 *****************************************************************************/

static int colloid_io_write_list_binary(colloid_io_t * cio,
					int ic, int jc, int kc, FILE * fp) {
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  colloids_info_cell_list_head(cio->info, ic, jc, kc, &pc);

  while (pc) {
    colloid_state_write_binary(pc->s, fp);
    pc = pc->next;
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_read_list_binary
 *
 *****************************************************************************/

static int colloid_io_read_list_binary(colloid_io_t * cio, int ndata,
				       FILE * fp) {

  int nread;
  colloid_state_t s;
  colloid_t * pc = NULL;

  assert(cio);
  assert(fp);

  for (nread = 0; nread < ndata; nread++) {
    colloid_state_read_binary(&s, fp);
    colloids_info_add_local(cio->info, s.index, s.r, &pc);
    if (pc)  pc->s = s;
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_ascii_set
 *
 *****************************************************************************/

int colloid_io_format_input_ascii_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_header_read = colloid_io_read_header_ascii;
  cio->f_list_read   = colloid_io_read_list_ascii;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_binary_set
 *
 *****************************************************************************/

int colloid_io_format_input_binary_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_header_read = colloid_io_read_header_binary;
  cio->f_list_read   = colloid_io_read_list_binary;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_input_serial_set
 *
 *****************************************************************************/

int colloid_io_format_input_serial_set(colloid_io_t * cio) {

  assert(cio);

  cio->single_file_read = 1;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_ascii_set
 *
 *****************************************************************************/

int colloid_io_format_output_ascii_set(colloid_io_t * cio) {

  assert(cio);

  cio->f_header_write = colloid_io_write_header_ascii;
  cio->f_list_write   = colloid_io_write_list_ascii;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_format_output_binary_set
 *
 *****************************************************************************/

int colloid_io_format_output_binary_set(colloid_io_t * cio) {

  cio->f_header_write = colloid_io_write_header_binary;
  cio->f_list_write   = colloid_io_write_list_binary;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_filename
 *
 *  Add the extension to the supplied stub.
 *
 *****************************************************************************/

static int colloid_io_filename(colloid_io_t * cio, char * filename,
			       const char * stub) {
  assert(cio);
  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* Check stub not too long */

  if (cio->index >= 1000) fatal("Format botch for cio stub %s\n", stub);

  sprintf(filename, "%s.%3.3d-%3.3d", stub, cio->n_io, cio->index + 1);

  return 0;
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

static int colloid_io_check_read(colloid_io_t * cio, int ngroup) {

  int nlocal;
  int ntotal;

  assert(cio);

  colloids_info_nlocal(cio->info, &nlocal);

  MPI_Reduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, 0, cio->comm);

  if (cio->single_file_read) {
    /* Only the global total can be compared (ngroup is ntotal). */
    nlocal = ntotal;
    MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, cio->xcomm);
  }

  if (cio->rank == 0) {
    if (ntotal != ngroup) {
      verbose("Colloid I/O group %d\n", cio->index);
      verbose("Colloids in file: %d Got %d\n", ngroup, ntotal);
      fatal("Total number of colloids not consistent with file\n");
    }
  }

  colloids_info_ntotal_set(cio->info);
  colloids_info_ntotal(cio->info, &ntotal);
  info("Read a total of %d colloids from file\n", ntotal);

  return 0;
}
