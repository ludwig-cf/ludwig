/*****************************************************************************
 *
 *  io_impl_mpio.c
 *
 *  Read/write aggregated data buffers using MPI/IO.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "io_impl_mpio.h"

/* Function table */
static io_impl_vt_t vt_ = {
  (io_impl_free_ft)         io_impl_mpio_free,
  (io_impl_read_ft)         io_impl_mpio_read,
  (io_impl_write_ft)        io_impl_mpio_write,
  (io_impl_write_begin_ft)  io_impl_mpio_write_begin, 
  (io_impl_write_end_ft)    io_impl_mpio_write_end
};

static int io_impl_mpio_types_create(io_impl_mpio_t * io);

/*****************************************************************************
 *
 *  io_impl_mpio_create
 *
 *****************************************************************************/

int io_impl_mpio_create(const io_metadata_t * metadata,
			io_impl_mpio_t ** io) {

  io_impl_mpio_t * mpio = NULL;

  mpio = (io_impl_mpio_t *) calloc(1, sizeof(io_impl_mpio_t));
  if (mpio == NULL) goto err;

  io_impl_mpio_initialise(metadata, mpio);
  *io = mpio;

  return 0;

 err:
  return -1;
}

/*****************************************************************************
 *
 *  io_impl_mpio_free
 *
 *****************************************************************************/

int io_impl_mpio_free(io_impl_mpio_t ** io) {

  assert(io);
  assert(*io);

  io_impl_mpio_finalise(*io);
  free(*io);
  *io = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_initialise
 *
 *****************************************************************************/

int io_impl_mpio_initialise(const io_metadata_t * metadata,
			    io_impl_mpio_t * io) {
  assert(metadata);
  assert(io);

  *io = (io_impl_mpio_t) {0};

  io->super.impl = &vt_;
  io_aggregator_create(metadata->element, metadata->limits, &io->super.aggr);
  /* ALLOW FAILURE? */

  io->metadata = metadata;

  io->fh = MPI_FILE_NULL;
  io_impl_mpio_types_create(io);

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_finalise
 *
 *****************************************************************************/

int io_impl_mpio_finalise(io_impl_mpio_t * io) {

  assert(io);

  MPI_Type_free(&io->file);
  MPI_Type_free(&io->array);
  MPI_Type_free(&io->element);
  io_aggregator_free(&io->super.aggr);

  *io = (io_impl_mpio_t) {0};

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_types_create
 *
 *****************************************************************************/

static int io_impl_mpio_types_create(io_impl_mpio_t * io) {

  const io_element_t * element = &io->metadata->element;
  const io_subfile_t * subfile = &io->metadata->subfile;

  int nlocal[3] = {0};
  int offset[3] = {0};

  cs_nlocal(io->metadata->cs, nlocal);
  cs_nlocal_offset(io->metadata->cs, offset);

  MPI_Type_contiguous(element->count, element->datatype, &io->element);
  MPI_Type_commit(&io->element);

  {
    int zero3[3] = {0};  /* No halo */
    int starts[3] = {0}; /* Local offset in the file */
    starts[X] = offset[X] - subfile->offset[X];
    starts[Y] = offset[Y] - subfile->offset[Y];
    starts[Z] = offset[Z] - subfile->offset[Z];

    /* Local array with no halo, and the file structure */
    MPI_Type_create_subarray(subfile->ndims, nlocal, nlocal, zero3,
			     MPI_ORDER_C, io->element, &io->array);
    MPI_Type_create_subarray(subfile->ndims, subfile->sizes, nlocal, starts,
			     MPI_ORDER_C, io->element, &io->file);
  }

  MPI_Type_commit(&io->array);
  MPI_Type_commit(&io->file);

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_write
 *
 *  Synchronous MPI_File_write_all() per communicator.
 *
 *****************************************************************************/

int io_impl_mpio_write(io_impl_mpio_t * io, const char * filename) {

  assert(io);
  assert(filename);

  {
    MPI_Comm comm = io->metadata->comm;
    MPI_Info info = MPI_INFO_NULL;       /* PENDING from io_options_t */
    MPI_Offset disp = 0;
    int count = 1;

    MPI_File_open(comm, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE, info,
		  &io->fh);
    MPI_File_set_view(io->fh, disp, io->element, io->file, "native", info);
    MPI_File_write_all(io->fh, io->super.aggr->buf, count, io->array,
		       &io->status);
    MPI_File_close(&io->fh);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_read
 *
 *****************************************************************************/

int io_impl_mpio_read(io_impl_mpio_t * io, const char * filename) {

  assert(io);
  assert(filename);

  {
    MPI_Comm comm = io->metadata->comm;
    MPI_Info info = MPI_INFO_NULL;       /* PENDING an io_option_t */
    MPI_Offset disp = 0;
    int count = 1;

    MPI_File_open(comm, filename, MPI_MODE_RDONLY, info, &io->fh);
    MPI_File_set_view(io->fh, disp, io->element, io->file, "native", info);
    MPI_File_read_all(io->fh, io->super.aggr->buf, count, io->array,
		      &io->status);
    MPI_File_close(&io->fh);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_impl_mpio_write_begin
 *
 *****************************************************************************/

int io_impl_mpio_write_begin(io_impl_mpio_t * io, const char * filename) {

  assert(io);
  assert(filename);

  {
    MPI_Comm comm = io->metadata->comm;
    MPI_Info info = MPI_INFO_NULL;       /* PENDING from io_options_t */
    MPI_Offset disp = 0;
    int count = 1;

    MPI_File_open(comm, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE, info,
		  &io->fh);
    MPI_File_set_view(io->fh, disp, io->element, io->file, "native", info);
    MPI_File_write_all_begin(io->fh, io->super.aggr->buf, count, io->array);
  }

  return 0;
}


/*****************************************************************************
 *
 *  io_impl_mpio_write_end
 *
 *****************************************************************************/

int io_impl_mpio_write_end(io_impl_mpio_t * io) {

  assert(io);

  MPI_File_write_all_end(io->fh, io->super.aggr->buf, &io->status);

  return 0;
}
