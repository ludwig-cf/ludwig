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
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2007-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "coords_s.h"
#include "leesedwards.h"
#include "io_harness.h"

typedef struct io_decomposition_s io_decomposition_t;

struct io_decomposition_s {
  int n_io;         /* Total number of I/O groups (files) in decomposition */
  int index;        /* Index of this I/O group {0, 1, ...} */
  MPI_Comm xcomm;   /* Cross-communicator for same rank in different groups */
  MPI_Comm comm;    /* MPI communicator for this group */
  int rank;         /* Rank of this process in group communicator */
  int size;         /* Size of this group in processes */
  int ngroup[3];    /* Global I/O group topology XYZ */
  int coords[3];    /* Coordinates of this group in I/O topology XYZ */
  int nsite[3];     /* Size of file in lattice sites */
  int offset[3];    /* Offset of the file on the lattice */
};

struct io_info_s {
  pe_t * pe;
  cs_t * cs;
  io_decomposition_t * io_comm;
  io_format_enum_t output_format;
  size_t bytesize;                   /* Actual output per site */
  size_t bytesize_ascii;             /* ASCII line size */
  size_t bytesize_binary;            /* Binary record size */
  int nsites;                        /* No. sites this rank */
  int maxlocal;                      /* Max. no. sites per rank this group */
  int metadata_written;
  int processor_independent;
  int single_file_read;
  int report;                        /* Report time taken for output */
  char metadata_stub[FILENAME_MAX];
  char name[FILENAME_MAX];
  io_rw_cb_ft write_data;
  io_rw_cb_ft write_ascii;
  io_rw_cb_ft write_binary;
  io_rw_cb_ft read_data;
  io_rw_cb_ft read_ascii;
  io_rw_cb_ft read_binary;
};

static void io_set_group_filename(char *, const char *, io_info_t *);
static long int io_file_offset(int, int, io_info_t *);
static int io_decomposition_create(pe_t * pe, cs_t * cs, const int grid[3],
				   io_decomposition_t ** p);
static int io_decomposition_free(io_decomposition_t *);


int io_write_data_p(io_info_t * obj, const char * filename_stub, void * data);
int io_write_data_s(io_info_t * obj, const char * filename_stub, void * data);
int io_unpack_local_buf(io_info_t * obj, int mpi_sender, const char * buf,
			char * io_buf);

/*****************************************************************************
 *
 *  io_info_create
 *
 *  Retrun a pointer to a new io_info_t with specified arguments.
 *
 *****************************************************************************/

int io_info_create(pe_t * pe, cs_t * cs, io_info_arg_t * arg, io_info_t ** p) {

  io_info_t * info = NULL;

  assert(pe);
  assert(cs);
  assert(arg);
  assert(p);

  info = (io_info_t *) calloc(1, sizeof(io_info_t));
  assert(info);
  if (info == NULL) pe_fatal(pe, "Failed to allocate io_info_t struct\n");

  info->pe = pe;
  info->cs = cs;

  io_decomposition_create(pe, cs, arg->grid, &info->io_comm);
  io_info_set_processor_dependent(info);
  info->single_file_read = 0;

  /* Local rank and group counts */

  info->nsites = info->io_comm->nsite[X]*info->io_comm->nsite[Y]
    *info->io_comm->nsite[Z];

  MPI_Reduce(&info->nsites, &info->maxlocal, 1, MPI_INT, MPI_MAX, 0,
	     info->io_comm->comm);

  *p = info;

  return 0;
}


/*****************************************************************************
 *
 *  io_decomposition_create
 *
 *  Set up an io_decomposition object using its size.
 *
 *****************************************************************************/

static int io_decomposition_create(pe_t * pe, cs_t * cs, const int grid[3],
				   io_decomposition_t ** pobj) {

  int i, colour;
  int n, offset;
  int ntotal[3];
  int noffset[3];
  int mpisz[3];
  int mpicoords[3];
  io_decomposition_t * p = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);
  assert(pobj);

  p = (io_decomposition_t *) calloc(1, sizeof(io_decomposition_t));
  assert(p);
  if (p == NULL) pe_fatal(pe, "Failed to allocate io_decomposition_t\n");

  p->n_io = 1;

  cs_cart_comm(cs, &comm);
  cs_nlocal_offset(cs, noffset);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);
  cs_ntotal(cs, ntotal);

  for (i = 0; i < 3; i++) {
    if (mpisz[i] % grid[i] != 0) {
      pe_fatal(pe, "Bad I/O grid (dim %d)\n", i);
    }
    p->ngroup[i] = grid[i];
    p->n_io *= grid[i];
    p->coords[i] = grid[i]*mpicoords[i]/mpisz[i];

    /* Offset and size, allowing for non-uniform decomposition */
    offset = mpicoords[i] / (mpisz[i]/grid[i]);
    p->offset[i] = cs->listnoffset[i][offset];
    p->nsite[i] = 0;
    for (n = offset; n < offset + (mpisz[i]/grid[i]); n++) {
      p->nsite[i] += cs->listnlocal[i][n];
    }
  }

  colour = p->coords[X]
         + p->coords[Y]*grid[X]
         + p->coords[Z]*grid[X]*grid[Y];

  p->index = colour;

  MPI_Comm_split(comm, colour, cs_cart_rank(cs), &p->comm);
  MPI_Comm_rank(p->comm, &p->rank);
  MPI_Comm_size(p->comm, &p->size);

  /* Cross-communicator */

  MPI_Comm_split(comm, p->rank, cs_cart_rank(cs), &p->xcomm);

  *pobj = p;

  return 0;
}

/*****************************************************************************
 *
 *  io_set_group_filename
 *
 *  Build the file name for this I/O group from the stub provided.
 *
 *****************************************************************************/

static void io_set_group_filename(char * filename_io, const char * stub,
				  io_info_t * info) {

  assert(stub);
  assert(strlen(stub) < FILENAME_MAX/2);  /* stub should not be too long */
  assert(info);
  assert(info->io_comm);
  assert(info->io_comm->n_io < 1000);     /* format restriction ... */


  sprintf(filename_io, "%s.%3.3d-%3.3d", stub, info->io_comm->n_io,
	  info->io_comm->index + 1);

  if (info->single_file_read) {
    sprintf(filename_io, "%s.%3.3d-%3.3d", stub, 1, 1);
  }

  return;
}

/*****************************************************************************
 *
 *  io_decomposition_free
 *
 *****************************************************************************/

static int io_decomposition_free(io_decomposition_t * p) {

  assert(p);
  MPI_Comm_free(&p->comm);
  free(p);
 
  return 0;
}

/*****************************************************************************
 *
 *  io_info_free
 *
 *  Deallocate io_info_t struct.
 *
 *****************************************************************************/

int io_info_free(io_info_t * p) {

  assert(p);

  io_decomposition_free(p->io_comm);
  free(p);

  return 0;
}

/*****************************************************************************
 *
 *  io_info_set_name
 *
 *****************************************************************************/

void io_info_set_name(io_info_t * p, const char * name) {

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

void io_info_set_processor_dependent(io_info_t * p) {

  assert(p);
  p->processor_independent = 0;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_processor_independent
 *
 *****************************************************************************/

void io_info_set_processor_independent(io_info_t * p) {

  assert(p);
  p->processor_independent = 1;

  return;
}

/*****************************************************************************
 *
 *  io_info_set_bytesize
 *
 *****************************************************************************/

int io_info_set_bytesize(io_info_t * p, io_format_enum_t t, size_t size) {

  assert(p);

  switch (t) {
  case IO_FORMAT_ASCII:
    p->bytesize_ascii = size;
    break;
  case IO_FORMAT_BINARY:
    p->bytesize_binary = size;
    break;
  default:
    pe_fatal(p->pe, "Bad io format %d\n", t);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_file_offset
 *
 *  Compute the file offset required for processor decomposition
 *  indepenedent files.
 *
 *****************************************************************************/

static long int io_file_offset(int ic, int jc, io_info_t * info) {

  long int offset;
  int ifo, jfo, kfo;
  int ntotal[3];
  int noffset[3];

  assert(info);

  /* Work out the offset of local lattice site (ic, jc, kc=1) in the file */
  ifo = info->io_comm->offset[X] + ic - 1;
  jfo = info->io_comm->offset[Y] + jc - 1;
  kfo = info->io_comm->offset[Z];

  offset = (ifo*info->io_comm->nsite[Y]*info->io_comm->nsite[Z]
	  + jfo*info->io_comm->nsite[Z]
	  + kfo)*info->bytesize;

  /* Single file offset */

  if (info->single_file_read) {
    cs_ntotal(info->cs, ntotal);
    cs_nlocal_offset(info->cs, noffset);
    ifo = noffset[X] + ic - 1;
    jfo = noffset[Y] + jc - 1;
    kfo = noffset[Z];
    offset = info->bytesize*(ifo*ntotal[Y]*ntotal[Z] + jfo*ntotal[Z] + kfo);
  }

  return offset;
}

/*****************************************************************************
 *
 *  io_write_metadata
 *
 *****************************************************************************/

int io_write_metadata(io_info_t * info) {

  assert(info);

  io_write_metadata_file(info, info->metadata_stub);

  return 0;
}

/*****************************************************************************
 *
 *  io_write_metadata_file
 *
 *  This describes, in human-readable form, the contents of the set
 *  of 1 or more files produced by a call to io_write.
 *
 *****************************************************************************/

#define MY_BUFSIZ 128

int io_write_metadata_file(io_info_t * info, char * filename_stub) {

  FILE * fp_meta;
  char filename_io[2*FILENAME_MAX];
  char buf[MY_BUFSIZ], rbuf[MY_BUFSIZ];
  char filename[FILENAME_MAX];
  int nr;
  int nx, ny, nz;
  int mpisz[3], mpicoords[3];
  int nlocal[3], noff[3], ntotal[3];

  const int tag = 1293;

  int sz;
  int le_nplane;
  double le_uy;
  MPI_Status status;

  /* Every group writes a file, ie., the information stub and
   * the details of the local group which allow the output to
   * be unmangled. */

  assert(info);

  sz = pe_mpi_size(info->pe);

  cs_ntotal(info->cs, ntotal);
  cs_nlocal(info->cs, nlocal);
  cs_nlocal_offset(info->cs, noff);
  cs_cartsz(info->cs, mpisz);
  cs_cart_coords(info->cs, mpicoords);

  le_nplane = 0;
  le_uy = 0.0;

  io_set_group_filename(filename, filename_stub, info);
  sprintf(filename_io, "%s.meta", filename);

  /* Write local decomposition information to the buffer */

  nr = sprintf(buf, "%3d %3d %3d %3d %d %d %d %d %d %d", info->io_comm->rank,
	       mpicoords[X], mpicoords[Y], mpicoords[Z],
	       nlocal[X], nlocal[Y], nlocal[Z], noff[X], noff[Y], noff[Z]);
  assert(nr < MY_BUFSIZ);

  if (info->io_comm->rank > 0) {
    MPI_Send(buf, MY_BUFSIZ, MPI_CHAR, 0, tag, info->io_comm->comm);
  }
  else {
      
    /* Root: write the information stub */

    nx = info->io_comm->ngroup[X];
    ny = info->io_comm->ngroup[Y];
    nz = info->io_comm->ngroup[Z];

    fp_meta = fopen(filename_io, "w");
    if (fp_meta == NULL) pe_fatal(info->pe, "fopen(%s) failed\n", filename_io);

    fprintf(fp_meta, "Metadata for file set prefix:    %s\n", filename_stub);
    fprintf(fp_meta, "Data description:                %s\n", info->name);
    fprintf(fp_meta, "Data size per site (bytes):      %d\n",
	    (int) info->bytesize);
    fprintf(fp_meta, "is_bigendian():                  %d\n", is_bigendian());
    fprintf(fp_meta, "Number of processors:            %d\n", sz);
    fprintf(fp_meta, "Cartesian communicator topology: %d %d %d\n",
	    mpisz[X], mpisz[Y], mpisz[Z]);
    fprintf(fp_meta, "Total system size:               %d %d %d\n",
	    ntotal[X], ntotal[Y], ntotal[Z]);
    /* Lees Edwards hardwired until refactor LE code dependencies */
    fprintf(fp_meta, "Lees-Edwards planes:             %d\n", le_nplane);
    fprintf(fp_meta, "Lees-Edwards plane speed         %16.14f\n", le_uy);
    fprintf(fp_meta, "Number of I/O groups (files):    %d\n", nx*ny*nz);
    fprintf(fp_meta, "I/O communicator topology:       %d %d %d\n",
	    nx, ny, nz);
    fprintf(fp_meta, "Write order:\n");

    /* Local information at root, and then in turn ... */

    fprintf(fp_meta, "%s\n", buf);

    for (nr = 1; nr < info->io_comm->size; nr++) {
      MPI_Recv(rbuf, MY_BUFSIZ, MPI_CHAR, nr, tag, info->io_comm->comm,
	       &status);
      fprintf(fp_meta, "%s\n", rbuf);
    }

    if (ferror(fp_meta)) {
      perror("perror: ");
      pe_fatal(info->pe, "File error on writing %s\n", filename_io);
    }
    fclose(fp_meta);
  }

  info->metadata_written = 1;

  return 0;
}

/*****************************************************************************
 *
 *  io_remove_metadata
 *
 *  Largely to clean up after automated tests; usually want to keep!
 *
 *****************************************************************************/

int io_remove_metadata(io_info_t * obj, const char * file_stub) {

  char subdirectory[FILENAME_MAX/2];
  char filename[FILENAME_MAX];
  char filename_io[2*FILENAME_MAX];

  assert(obj);
  assert(file_stub);

  if (obj->io_comm->rank == 0) {
    pe_subdirectory(obj->pe, subdirectory);
    io_set_group_filename(filename, file_stub, obj);
    sprintf(filename_io, "%s%s.meta", subdirectory, filename);
    remove(filename_io);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_remove
 *
 *  Remove filename on each IO root.
 *
 *****************************************************************************/

int io_remove(const char * filename_stub, io_info_t * obj) {

  char subdirectory[FILENAME_MAX];
  char filename[FILENAME_MAX];

  assert(filename_stub);
  assert(obj);

  if (obj->io_comm->rank == 0) {
    pe_subdirectory(obj->pe, subdirectory);
    io_set_group_filename(filename, filename_stub, obj);
    remove(filename);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_info_format_set
 *
 *****************************************************************************/

int io_info_format_set(io_info_t * obj, int form_in, int form_out) {

  assert(obj);
  assert(form_in >= 0);
  assert(form_in <= IO_FORMAT_DEFAULT);
  assert(form_out >= 0);
  assert(form_out <= IO_FORMAT_DEFAULT);

  io_info_format_in_set(obj, form_in);
  io_info_format_out_set(obj, form_out);

  return 0;
}

/*****************************************************************************
 *
 *  io_info_format_in_set
 *
 *  Set input format.
 *
 *****************************************************************************/

int io_info_format_in_set(io_info_t * obj, int form_in) {

  assert(obj);
  assert(form_in >= 0);
  assert(form_in <= IO_FORMAT_DEFAULT);

  if (form_in == IO_FORMAT_NULL) return 0;

  switch (form_in) {
  case IO_FORMAT_ASCII_SERIAL:
    obj->read_data = obj->read_ascii;
    obj->processor_independent = 1;
    break;
  case IO_FORMAT_BINARY_SERIAL:
    obj->read_data = obj->read_binary;
    obj->processor_independent = 1;
    obj->single_file_read = 1;
    break;
  case IO_FORMAT_ASCII:
    obj->read_data = obj->read_ascii;
    obj->processor_independent = 0;
    break;
  case IO_FORMAT_BINARY:
  case IO_FORMAT_DEFAULT:
    obj->read_data = obj->read_binary;
    obj->processor_independent = 1;
    obj->single_file_read = 1;
    break;
  default:
    pe_fatal(obj->pe, "Bad i/o input format\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_info_format_out_set
 *
 *****************************************************************************/

int io_info_format_out_set(io_info_t * obj, int form_out) {

  assert(obj);
  assert(form_out >= 0);
  assert(form_out <= IO_FORMAT_DEFAULT);

  if (form_out == IO_FORMAT_NULL) return 0;

  switch (form_out) {
  case IO_FORMAT_ASCII:
    obj->output_format = IO_FORMAT_ASCII;
    obj->write_data = obj->write_ascii;
    obj->processor_independent = 0;
    obj->bytesize = obj->bytesize_ascii;
    break;
  case IO_FORMAT_BINARY:
  case IO_FORMAT_DEFAULT:
    obj->output_format = IO_FORMAT_BINARY;
    obj->write_data = obj->write_binary;
    obj->processor_independent = 1;
    obj->bytesize = obj->bytesize_binary;
    break;
  default:
    pe_fatal(obj->pe, "Bad i/o output format\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_info_read_set
 *
 *****************************************************************************/

int io_info_read_set(io_info_t * obj, int format, io_rw_cb_ft f) {

  assert(obj);
  assert(format == IO_FORMAT_ASCII || format == IO_FORMAT_BINARY);
  assert(f);

  if (format == IO_FORMAT_ASCII) obj->read_ascii = f;
  if (format == IO_FORMAT_BINARY) obj->read_binary = f;

  return 0;
}

/*****************************************************************************
 *
 *  io_info_write_set
 *
 *****************************************************************************/

int io_info_write_set(io_info_t * obj, int format, io_rw_cb_ft f) {

  assert(obj);
  assert(format == IO_FORMAT_ASCII || format == IO_FORMAT_BINARY);
  assert(f);

  if (format == IO_FORMAT_ASCII) obj->write_ascii = f;
  if (format == IO_FORMAT_BINARY) obj->write_binary = f;

  return 0;
}

/*****************************************************************************
 *
 *  io_write_data
 *
 *  This is the driver to write lattice quantities on the lattice.
 *  The arguments are the filename stub and the io_info struct
 *  describing which quantity we are dealing with.
 *
 *  The third argument is an opaque pointer to the data object,
 *  which will be passed to the callback which does the write.
 *
 *****************************************************************************/

int io_write_data(io_info_t * obj, const char * filename_stub, void * data) {

  double t0, t1;

  assert(obj);
  assert(data);

  if (obj->processor_independent == 0) {
    /* Use the standard "parallel" method for the time being. */
    io_write_data_p(obj, filename_stub, data);
  }
  else {
    /* This is serial output format if one I/O group */
    assert(obj->io_comm->ngroup[X] == 1);
    t0 = MPI_Wtime();
    io_write_data_s(obj, filename_stub, data);
    t1 = MPI_Wtime();
    if (obj->report) {
      pe_info(obj->pe, "Write %lu bytes in %f secs %f GB/s\n",
	      obj->nsites*obj->bytesize, t1-t0,
	      obj->nsites*obj->bytesize/(1.0e+09*(t1-t0)));
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_write_p
 *
 *  All writes are processor decomposition dependent in this
 *  version, which also has a potentially unsafe fopen("a").
 *
 *****************************************************************************/

int io_write_data_p(io_info_t * obj, const char * filename_stub, void * data) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  int       token = 0;
  int       ic, jc, kc, index;
  int       nlocal[3];
  const int io_tag = 140;

  MPI_Status status;

  assert(obj);
  assert(data);
  assert(obj->write_data);

  if (obj->metadata_written == 0) io_write_metadata(obj);

  cs_nlocal(obj->cs, nlocal);
  io_set_group_filename(filename_io, filename_stub, obj);

  if (obj->io_comm->rank == 0) {
    /* Open the file anew */
    fp_state = fopen(filename_io, "wb");
  }
  else {

    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can re-open the file and write
     * our own data. */

    MPI_Recv(&token, 1, MPI_INT, obj->io_comm->rank - 1, io_tag,
	     obj->io_comm->comm, &status);
    fp_state = fopen(filename_io, "ab");
  }

  if (fp_state == NULL) pe_fatal(obj->pe, "Failed to open %s\n", filename_io);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = cs_index(obj->cs, ic, jc, kc);
	obj->write_data(fp_state, index, data);
      }
    }
  }

  /* Check the error indicator on the stream and close */

  if (ferror(fp_state)) {
    perror("perror: ");
    pe_fatal(obj->pe, "File error on writing %s\n", filename_io);
  }
  fclose(fp_state);

  /* Pass the token to the next process to write */

  if (obj->io_comm->rank < obj->io_comm->size - 1) {
    MPI_Ssend(&token, 1, MPI_INT, obj->io_comm->rank + 1, io_tag,
	      obj->io_comm->comm);
  }

  return 0;
}

/******************************************************************************
 *
 *  io_write_data_s
 *
 *  Write data to file in decomposition-independent fashion. Data
 *  are aggregated to a contiguous buffer internally, and tranferred
 *  to a single block at rank 0 per I/O group before output to file.
 *
 *****************************************************************************/

/* TODO */
/* Write I/O documentation */
/* Should really have separate processor_independent / single_file for I/O */
/* Appropriate switch in io-write-data */
/* Meta data files need to match actual output format */
/* Add flag for report timings */

int io_write_data_s(io_info_t * obj, const char * filename_stub, void * data) {

  int nr;
  int ic, jc, kc, index;
  int nlocal[3];
  int itemsz;                      /* Data size per site (bytes) */
  int iosz;                        /* Data size io_buf (bytes) */
  int localsz;                     /* Data size local buffer (bytes) */
  char * buf = NULL;               /* Local buffer for this rank */
  char * io_buf = NULL;            /* I/O buffer for whole group */
  char * rbuf = NULL;              /* Recv buffer */
  char filename_io[FILENAME_MAX];
  long int offset;
  FILE * fp_state = NULL;
  FILE * fp_buf;

  const int tag = 2017;
  MPI_Status status;

  assert(obj);
  assert(data);
  assert(obj->write_data);

  if (obj->metadata_written == 0) io_write_metadata(obj);

  cs_nlocal(obj->cs, nlocal);
  io_set_group_filename(filename_io, filename_stub, obj);
  sprintf(filename_io, "%s.%3.3d-%3.3d", filename_stub, 1, 1);

  itemsz = obj->bytesize;

  /* Local buffer to be assoicated with file handle for write... */

  localsz = itemsz*nlocal[X]*nlocal[Y]*nlocal[Z];
  buf = (char *) malloc(localsz*sizeof(char));
  fp_buf = fopen("/dev/null", "w"); /* TODO: de-hardwire this */
  setvbuf(fp_buf, buf, _IOFBF, localsz);

  if (buf == NULL || fp_buf == NULL) {
    pe_fatal(obj->pe, "Buffer initialisation failed\n");
  }

  /* Write to the local buffer in local order */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = cs_index(obj->cs, ic, jc, kc);
	obj->write_data(fp_buf, index, data);
      }
    }
  }

  /* Send local buffer to root. */

  if (obj->io_comm->rank > 0) {
    MPI_Send(buf, localsz, MPI_BYTE, 0, tag, obj->io_comm->comm);
  }
  else {

    /* I/O buff */

    iosz = itemsz*obj->nsites;
    io_buf = (char *) malloc(iosz*sizeof(char));
    if (io_buf == NULL) pe_fatal(obj->pe, "malloc(io_buf)\n");

    rbuf = (char *) malloc(itemsz*obj->maxlocal*sizeof(char));
    if (rbuf == NULL) pe_fatal(obj->pe, "malloc(rbuf)");

    /* Unpack own buffer to correct position in the io buffer, and
     * then do it for incoming messages. */

    io_unpack_local_buf(obj, 0, buf, io_buf);

    for (nr = 0; nr < obj->io_comm->size - 1; nr++) {
      MPI_Recv(rbuf, itemsz*obj->maxlocal, MPI_BYTE, MPI_ANY_SOURCE, tag,
	       obj->io_comm->comm, &status);
      io_unpack_local_buf(obj, status.MPI_SOURCE, rbuf, io_buf);
    }

    free(rbuf);

    /* Write the file for this group. Group zero creates the file
     * before allowing other groups to write at appropriate offset. */

    if (obj->io_comm->index == 0) {
      fp_state = fopen(filename_io, "w");
    }

    MPI_Bcast(&itemsz, 1, MPI_INT, 0, obj->io_comm->xcomm);

    if (obj->io_comm->index > 0) {
      fp_state = fopen(filename_io, "r+");
      offset = obj->io_comm->offset[X]*
	       obj->io_comm->nsite[Y]*obj->io_comm->nsite[Z];
      fseek(fp_state, offset*itemsz, SEEK_SET);
    }

    if (fp_state == NULL) {
      pe_fatal(obj->pe, "Failed to open %s\n", filename_io);
    }

    fwrite(io_buf, sizeof(char), iosz, fp_state);

    if (ferror(fp_state)) {
      perror("perror: ");
      pe_fatal(obj->pe, "File error on writing %s\n", filename_io);
    }
    fclose(fp_state);
    free(io_buf);
  }

  fclose(fp_buf);
  free(buf);

  return 0;
}

/****************************************************************************
 *
 *  io_unpack_local_buf
 *
 *  Used in aggregating data to the group output buffer for decomposition
 *  indenpendent I/O.
 *
 *  Copy data (buf) from the sender (in the io group communicator) to the
 *  group buffer (io_buf).
 *
 ****************************************************************************/ 

int io_unpack_local_buf(io_info_t * obj, int mpi_sender, const char * buf,
			char * io_buf) {
  int ib = 0;
  int rank;
  int offset;
  int ic, jc;
  int itemsz;
  int coords[3];
  int nsendlocal[3];
  int nsendoffset[3];
  int ifo, jfo, kfo;
  MPI_Comm cartcomm;
  MPI_Group iogrp, cartgrp;

  assert(obj);
  assert(buf);
  assert(io_buf);

  cs_cart_comm(obj->cs, &cartcomm);

  /* We need to work out nlocal and noffset in the sender
   * via the Cartesian coordinates of the sender. */

  MPI_Comm_group(obj->io_comm->comm, &iogrp);
  MPI_Comm_group(cartcomm, &cartgrp);
  MPI_Group_translate_ranks(iogrp, 1, &mpi_sender, cartgrp, &rank);
  MPI_Cart_coords(obj->cs->commcart, rank, 3, coords);

  for (ic = 0; ic < 3; ic++) {
    nsendlocal[ic] = obj->cs->listnlocal[ic][coords[ic]];
    nsendoffset[ic] = obj->cs->listnoffset[ic][coords[ic]];
  }

  itemsz = obj->bytesize;

  /* Copy the local data into the correct position in the group
   * buffer. Contiguous strips in z-direction. */

  for (ic = 1; ic <= nsendlocal[X]; ic++) {
    for (jc = 1; jc <= nsendlocal[Y]; jc++) {
      ifo = (nsendoffset[X] + ic - 1) - obj->io_comm->offset[X];
      jfo = (nsendoffset[Y] + jc - 1) - obj->io_comm->offset[Y];
      kfo = nsendoffset[Z] - obj->io_comm->offset[Z];

      offset = ifo*obj->io_comm->nsite[Y]*obj->io_comm->nsite[Z]
	+ jfo*obj->io_comm->nsite[Z] + kfo;

      memcpy(io_buf + itemsz*offset, buf + ib, itemsz*nsendlocal[Z]);
      ib += itemsz*nsendlocal[Z];
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  io_read_data
 *
 *  Driver for reads.
 *
 *****************************************************************************/

int io_read_data(io_info_t * obj, const char * filename_stub, void * data) {

  FILE *    fp_state;
  char      filename_io[FILENAME_MAX];
  long int  token = 0;
  int       ic, jc, kc, index;
  int       nlocal[3];
  long int  offset;
  const int io_tag = 141;

  MPI_Status status;

  assert(obj);
  assert(obj->read_data);
  assert(filename_stub);
  assert(data);

  cs_nlocal(obj->cs, nlocal);

  io_set_group_filename(filename_io, filename_stub, obj);

  if (obj->io_comm->rank == 0) {

    fp_state = fopen(filename_io, "r");
  }
  else {

    /* Non-io-root process. Block until we receive the token from the
     * previous process, after which we can re-open the file and read. */

    MPI_Recv(&token, 1, MPI_LONG, obj->io_comm->rank - 1, io_tag,
	     obj->io_comm->comm, &status);
    fp_state = fopen(filename_io, "r");
  }

  if (fp_state == NULL) pe_fatal(obj->pe, "Failed to open %s\n", filename_io);
  fseek(fp_state, token, SEEK_SET);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      /* Work out where the read comes from if required */
      offset = io_file_offset(ic, jc, obj);
      if (obj->processor_independent) fseek(fp_state, offset, SEEK_SET);

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = cs_index(obj->cs, ic, jc, kc);
	obj->read_data(fp_state, index, data);
      }
    }
  }

  /* The token is the current offset for processor-dependent output */

  token = ftell(fp_state);

  /* Check the error indicator on the stream and close */

  if (ferror(fp_state)) {
    perror("perror: ");
    pe_fatal(obj->pe, "File error on reading %s\n", filename_io);
  }
  fclose(fp_state);

  /* Pass the token to the next process to read */

  if (obj->io_comm->rank < obj->io_comm->size - 1) {
    MPI_Ssend(&token, 1, MPI_LONG, obj->io_comm->rank + 1, io_tag,
	      obj->io_comm->comm);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_info_single_file_set
 *
 *****************************************************************************/

void io_info_single_file_set(io_info_t * info) {

  assert(info);

  info->single_file_read = 1;

  return;
}

/*****************************************************************************
 *
 *  io_info_metadata_filestub_set
 *
 *****************************************************************************/

int io_info_metadata_filestub_set(io_info_t * info, const char * stub) {

  assert(info);
  assert(stub);
  assert(strlen(stub) < FILENAME_MAX);

  strncpy(info->metadata_stub, stub, FILENAME_MAX);

  return 0;
}
