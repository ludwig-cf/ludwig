/*****************************************************************************
 *
 *  io_harness.c
 *
 *  Drivers for serial/parallel IO for lattice quantities.
 *
 *  Each quantity (e.g., distributions, order parameter) stored on
 *  the lattice should set up an io_info struct which tells the
 *  io_harness how to actually do the read/write.
 *
 *  Actual read and writes can be initiated with a call io_read() or
 *  io_write() with the appropriate io_info struct.
 *
 *  Parallel IO takes place by taking a Cartesian decomposition of
 *  the system which can be the same, or coarser than that of the
 *  lattice Cartesian communicator. Each IO communicator group so
 *  defined then deals with its own file.
 *
 *  $Id: io_harness.c,v 1.3 2008-11-04 16:46:44 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "runtime.h"
#include "io_harness.h"

struct io_decomposition_t {
  int n_io;         /* Total number of I/O groups (files) in decomposition */
  int index;        /* Index of this I/O group {0, 1, ...} */
  MPI_Comm comm;    /* MPI communicator for this group */
  int rank;         /* Rank of this process in communicator */
  int size;         /* Size of this group in processes */
  int ngroup[3];    /* Global I/O group topology XYZ */
  int coords[3];    /* Coordinates of this group in I/O topology XYZ */
  int nsite[3];     /* Size of file in lattice sites */
  int offset[3];    /* Offset of the file on the lattice */
};

struct io_info_t {
  struct io_decomposition_t * io_comm;
  size_t bytesize;
  int processor_independent;
  char name[FILENAME_MAX];
  int (* write_function)   (FILE *, const int, const int, const int);
  int (* read_function)    (FILE *, const int, const int, const int);
  int (* write_function_a) (FILE *, const int, const int, const int);
  int (* read_function_a)  (FILE *, const int, const int, const int);
  int (* write_function_b) (FILE *, const int, const int, const int);
  int (* read_function_b)  (FILE *, const int, const int, const int);
};

static struct io_info_t * io_info_allocate(void);
static void io_set_group_filename(char *, const char *, struct io_info_t *);
static long int io_file_offset(int, int, struct io_info_t *);
static struct io_decomposition_t * io_decomposition_allocate(void);
static struct io_decomposition_t * io_decomposition_create(const int grid[3]);
static void io_decomposition_destroy(struct io_decomposition_t *);

/*****************************************************************************
 *
 *  io_info_create
 *
 *  Return a pointer to an io_info object with default decomposition.
 *
 *****************************************************************************/

struct io_info_t * io_info_create() {

  int io_grid[3] = {1, 1, 1}; /* Default i/o grid */
  struct io_info_t * p_info;

  RUN_get_int_parameter_vector("io_grid", io_grid); /* User input grid */
  p_info = io_info_create_with_grid(io_grid);

  return p_info;
}

/*****************************************************************************
 *
 *  io_info_create_with_grid
 *
 *  Retrun a pointer to a new io_info object with specified decomposition.
 *
 *****************************************************************************/

struct io_info_t * io_info_create_with_grid(const int grid[3]) {

  struct io_info_t * p_info;
  struct io_decomposition_t * p_decomp;

  p_info = io_info_allocate();
  p_decomp = io_decomposition_create(grid);
  p_info->io_comm = p_decomp;
  io_info_set_processor_dependent(p_info);

  return p_info;
}

/*****************************************************************************
 *
 *  io_decomposition_create
 *
 *  Set up an io_decomposition object using its size.
 *
 *****************************************************************************/

static struct io_decomposition_t * io_decomposition_create(const int grid[3]) {

  int i, colour;
  int noffset[3];
  struct io_decomposition_t * p = NULL;
  MPI_Comm comm = cart_comm();

  assert(comm != MPI_COMM_NULL);
  get_N_offset(noffset);

  p = io_decomposition_allocate();
  p->n_io = 1;

  for (i = 0; i < 3; i++) {
    if (cart_size(i) % grid[i] != 0) fatal("Bad I/O grid (dim %d)\n", i);
    p->ngroup[i] = grid[i];
    p->n_io *= grid[i];
    p->coords[i] = grid[i]*cart_coords(i)/cart_size(i);
    p->nsite[i] = N_total(i)/grid[i];
    p->offset[i] = noffset[i] - p->coords[i]*p->nsite[i];
  }

  colour = p->coords[X]
         + p->coords[Y]*grid[X]
         + p->coords[Z]*grid[X]*grid[Y];

  p->index = colour;

  MPI_Comm_split(comm, colour, cart_rank(), &p->comm);
  MPI_Comm_rank(p->comm, &p->rank);
  MPI_Comm_size(p->comm, &p->size);

  return p;
}

/*****************************************************************************
 *
 *  io_write
 *
 *  This is the driver to write lattice quantities on the lattice.
 *  The arguments are the filename stub and the io_info struct
 *  describing which quantity we are dealing with.
 *
 *  All writes are processor decomposition dependent at the moment.
 *
 *****************************************************************************/

void io_write(char * filename_stub, struct io_info_t * io_info) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  int       token = 0;
  int       ic, jc, kc;
  int       nlocal[3];
  const int io_tag = 140;

  MPI_Status status;

  assert(io_info);
  assert(io_info->write_function);

  get_N_local(nlocal);
  io_set_group_filename(filename_io, filename_stub, io_info);

  if (io_info->io_comm->rank == 0) {
    /* Open the file anew */
    fp_state = fopen(filename_io, "wb");
  }
  else {

    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can re-open the file and write
     * our own data. */

    MPI_Recv(&token, 1, MPI_INT, io_info->io_comm->rank - 1, io_tag,
	     io_info->io_comm->comm, &status);
    fp_state = fopen(filename_io, "ab");
  }

  if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	io_info->write_function(fp_state, ic, jc, kc);
      }
    }
  }

  /* Check the error indicator on the stream and close */

  if (ferror(fp_state)) {
    perror("perror: ");
    fatal("File error on writing %s\n", filename_io);
  }
  fclose(fp_state);

  /* Pass the token to the next process to write */

  if (io_info->io_comm->rank < io_info->io_comm->size - 1) {
    MPI_Ssend(&token, 1, MPI_INT, io_info->io_comm->rank + 1, io_tag,
	      io_info->io_comm->comm);
  }

  return;
}

/*****************************************************************************
 *
 *  io_read
 *
 *  Driver for reads. Comments as for io_write.
 *
 *****************************************************************************/

void io_read(char * filename_stub, struct io_info_t * io_info) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  long int  token = 0;
  int       ic, jc, kc;
  int       nlocal[3];
  long int  offset;
  const int io_tag = 141;

  MPI_Status status;

  assert(io_info);

  get_N_local(nlocal);

  io_set_group_filename(filename_io, filename_stub, io_info);

  if (io_info->io_comm->rank == 0) {

    fp_state = fopen(filename_io, "r");
  }
  else {

    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can re-open the file and read. */

    MPI_Recv(&token, 1, MPI_LONG, io_info->io_comm->rank - 1, io_tag,
	     io_info->io_comm->comm, &status);
    fp_state = fopen(filename_io, "r");
  }

  if (fp_state == NULL) fatal("Failed to open %s\n", filename_io);
  fseek(fp_state, token, SEEK_SET);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      /* Work out where the read comes from if required */
      offset = io_file_offset(ic, jc, io_info);
      if (io_info->processor_independent) fseek(fp_state, offset, SEEK_SET);

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	io_info->read_function(fp_state, ic, jc, kc);
      }
    }
  }

  /* The token is the current offset for processor-dependent output */

  token = ftell(fp_state);

  /* Check the error indicator on the stream and close */

  if (ferror(fp_state)) {
    perror("perror: ");
    fatal("File error on reading %s\n", filename_io);
  }
  fclose(fp_state);

  /* Pass the token to the next process to read */

  if (io_info->io_comm->rank < io_info->io_comm->size - 1) {
    MPI_Ssend(&token, 1, MPI_LONG, io_info->io_comm->rank + 1, io_tag,
	      io_info->io_comm->comm);
  }

  return;
}

/*****************************************************************************
 *
 *  io_set_group_filename
 *
 *  Build the file name for this I/O group from the stub provided.
 *
 *****************************************************************************/

static void io_set_group_filename(char * filename_io, const char * stub,
				  struct io_info_t * info) {

  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* stub should not be too long */
  assert(info);
  assert(info->io_comm);
  assert(info->io_comm->n_io < 1000);     /* format restriction ... */


  sprintf(filename_io, "%s.%3.3d-%3.3d", stub, info->io_comm->n_io,
	  info->io_comm->index + 1);

  return;
}

/*****************************************************************************
 *
 *  io_decomposition_allocate
 *
 *  Allocate an io_decomposition_t object or fail gracefully.
 *
 *****************************************************************************/

static struct io_decomposition_t * io_decomposition_allocate() {

  struct io_decomposition_t * p;

  p = (struct io_decomposition_t *) malloc(sizeof(struct io_decomposition_t));
  if (p == NULL) fatal("Failed to allocate io_decomposition_t\n");

  return p;
}

/*****************************************************************************
 *
 *  io_decomposition_destroy
 *
 *****************************************************************************/

static void io_decomposition_destroy(struct io_decomposition_t * p) {

  assert(p);
  MPI_Comm_free(&p->comm);
  free(p);
 
  return;
}

/*****************************************************************************
 *
 *  io_info_allocate
 *
 *  Return a pointer to newly allocated io_info object.
 *
 *****************************************************************************/

struct io_info_t * io_info_allocate() {

  struct io_info_t * p;

  p = (struct io_info_t *) malloc(sizeof(struct io_info_t));
  if (p == NULL) fatal("Failed to allocate io_info_t struct\n");

  return p;
}

/*****************************************************************************
 *
 *  io_info_destroy
 *
 *  Deallocate io_info_t struct.
 *
 *****************************************************************************/

void io_info_destroy(struct io_info_t * p) {

  assert(p != (struct io_info_t *) NULL);
  io_decomposition_destroy(p->io_comm);
  free(p);

  return;
}

/*****************************************************************************
 *
 *  io_info_set_write
 *
 *****************************************************************************/

void io_info_set_write(struct io_info_t * p,
		       int (* writer) (FILE *, int, int, int)) {

  assert(p != (struct io_info_t *) NULL);
  p->write_function = writer;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_read
 *
 *****************************************************************************/

void io_info_set_read(struct io_info_t * p,
		      int (* reader) (FILE *, int, int, int)) {

  assert(p);
  p->read_function = reader;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_write_ascii
 *
 *****************************************************************************/

void io_info_set_write_ascii(struct io_info_t * p,
			     int (* writer) (FILE *, int, int, int)) {

  assert(p);
  p->write_function_a = writer;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_read_ascii
 *
 *****************************************************************************/

void io_info_set_read_ascii(struct io_info_t * p,
			    int (* reader) (FILE *, int, int, int)) {

  assert(p);
  p->read_function_a = reader;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_write_binary
 *
 *****************************************************************************/

void io_info_set_write_binary(struct io_info_t * p,
			      int (* writer) (FILE *, int, int, int)) {

  assert(p);
  p->write_function_b = writer;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_read_binary
 *
 *****************************************************************************/

void io_info_set_read_binary(struct io_info_t * p,
			     int (* reader) (FILE *, int, int, int)) {
  assert(p);
  p->read_function_b = reader;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_format_ascii
 *
 *****************************************************************************/

void io_info_set_format_ascii(struct io_info_t * p) {

  assert(p);
  assert(p->processor_independent == 0);

  p->read_function = p->read_function_a;
  p->write_function = p->write_function_a;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_format_binary
 *
 *****************************************************************************/

void io_info_set_format_binary(struct io_info_t * p) {

  assert(p);
  p->read_function = p->read_function_b;
  p->write_function = p->write_function_b;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_name
 *
 *****************************************************************************/

void io_info_set_name(struct io_info_t * p, const char * name) {

  assert(p);
  assert(strlen(name) < FILENAME_MAX);
  strcpy(p->name, name);

  return;
}

/*****************************************************************************
 *
 *  io_info_set_processor_dependent
 *
 *****************************************************************************/

void io_info_set_processor_dependent(struct io_info_t * p) {

  assert(p);
  p->processor_independent = 0;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_processor_independent
 *
 *****************************************************************************/

void io_info_set_processor_independent(struct io_info_t * p) {

  assert(p);
  p->processor_independent = 1;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_bytesize
 *
 *****************************************************************************/

void io_info_set_bytesize(struct io_info_t * p, size_t size) {

  assert(p);
  p->bytesize = size;

  return;
}

/*****************************************************************************
 *
 *  io_file_offset
 *
 *  Compute the file offset required for processor decomposition
 *  indepenedent files.
 *
 *****************************************************************************/

static long int io_file_offset(int ic, int jc, struct io_info_t * info) {

  long int offset;
  int ifo, jfo, kfo;

  /* Work out the offset of local lattice site (ic, jc, kc=1) in the file */
  ifo = info->io_comm->offset[X] + ic - 1;
  jfo = info->io_comm->offset[Y] + jc - 1;
  kfo = info->io_comm->offset[Z];

  offset = (ifo*info->io_comm->nsite[Y]*info->io_comm->nsite[Z]
	  + jfo*info->io_comm->nsite[Z]
	  + kfo)*info->bytesize;

  return offset;
}

/*****************************************************************************
 *
 *  io_write_metadata
 *
 *  This describes, in human-readable form, the contents of the set
 *  of 1 or more files produced by a call to io_write.
 *
 *****************************************************************************/

void io_write_metadata(char * filename_stub, struct io_info_t * info) {

  FILE * fp_meta;
  char filename_io[FILENAME_MAX];
  int  nx, ny, nz;
  int n[3], noff[3];

  int token = 0;
  const int tag = 1293;
  MPI_Status status;

  /* Every group writes a file, ie., the information stub and
   * the details of the local group which allow the output to
   * be unmangled. */

  assert(info);
  get_N_offset(noff);
  
  io_set_group_filename(filename_io, filename_stub, info);
  sprintf(filename_io, "%s.meta", filename_io);


  if(info->io_comm->rank == 0) {
    /* Write the information stub */

    nx = info->io_comm->ngroup[X];
    ny = info->io_comm->ngroup[Y];
    nz = info->io_comm->ngroup[Z];

    fp_meta = fopen(filename_io, "w");
    if (fp_meta == NULL) fatal("fopen(%s) failed\n", filename_io);

    fprintf(fp_meta, "Metadata for file set prefix:    %s\n", filename_stub);
    fprintf(fp_meta, "Data description:                %s\n", info->name);
    fprintf(fp_meta, "Data size per site (bytes):      %d\n",
	    (int) info->bytesize);
    fprintf(fp_meta, "is_bigendian():                  %d\n", is_bigendian());
    fprintf(fp_meta, "Number of processors:            %d\n", pe_size());
    fprintf(fp_meta, "Cartesian communicator topology: %d %d %d\n",
	    cart_size(X), cart_size(Y), cart_size(Z));
    fprintf(fp_meta, "Total system size:               %d %d %d\n",
	    N_total(X), N_total(Y), N_total(Z));
    /* Lees Edwards hardwired until refactor LE code dependencies */
    fprintf(fp_meta, "Lees-Edwards planes:             %d\n", 0);
    fprintf(fp_meta, "Lees-Edwards plane speed         %16.14f\n", 0.0);
    fprintf(fp_meta, "Number of I/O groups (files):    %d\n", nx*ny*nz);
    fprintf(fp_meta, "I/O communicator topology:       %d %d %d\n",
	    nx, ny, nz);
    fprintf(fp_meta, "Write order:\n");

  }
  else {
    MPI_Recv(&token, 1, MPI_INT, info->io_comm->rank - 1, tag,
	     info->io_comm->comm, &status);
    fp_meta = fopen(filename_io, "a");
    if (fp_meta == NULL) fatal("fopen(%s) failed\n", filename_io);
  }

  /* Local decomposition information */

  get_N_local(n);
  fprintf(fp_meta, "%3d %3d %3d %3d %d %d %d %d %d %d\n", info->io_comm->rank,
          cart_coords(X), cart_coords(Y), cart_coords(Z),
          n[X], n[Y], n[Z], noff[X], noff[Y], noff[Z]);

  if (ferror(fp_meta)) {
    perror("perror: ");
    fatal("File error on writing %s\n", filename_io);
  }
  fclose(fp_meta);

 if(info->io_comm->rank < info->io_comm->size - 1) {
   MPI_Ssend(&token, 1, MPI_INT, info->io_comm->rank + 1, tag,
	     info->io_comm->comm);
 }

  return;
}
