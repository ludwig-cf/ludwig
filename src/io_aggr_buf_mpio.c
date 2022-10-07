/*****************************************************************************
 *
 *  io_aggr_buf_mpio.c
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

#include "io_aggr_buf_mpio.h"

/*****************************************************************************
 *
 *  io_aggr_buf_write
 *
 *  Generalise to case where separate write_all in comm.
 *
 *****************************************************************************/

int io_aggr_mpio_write(pe_t * pe, cs_t * cs, const char * filename,
		       const io_aggr_buf_t * buf) {
  assert(pe);
  assert(cs);
  assert(filename);
  assert(buf);

  int ndims = 3;          /* nz may be 1; ie., 2-dimensional system */
  int sizes[3] = {0};     /* ie., ntotal */
  int subsizes[3] = {0};  /* ie., nlocal */
  int starts[3] = {0};    /* ie., local offset */
  int zero3[3] = {0};

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Datatype etype = MPI_DATATYPE_NULL;     /* element description */
  MPI_Datatype array = MPI_DATATYPE_NULL;     /* local array description */
  MPI_Datatype filetype = MPI_DATATYPE_NULL;  /* global description */

  cs_cart_comm(cs, &comm);
  cs_ntotal(cs, sizes);
  cs_nlocal(cs, subsizes);
  cs_nlocal_offset(cs, starts);

  /* Element type (multiple of MPI_CHAR), and file type */

  MPI_Type_contiguous(buf->szelement, MPI_CHAR, &etype);
  MPI_Type_create_subarray(ndims, subsizes, subsizes, zero3, MPI_ORDER_C,
			   etype, &array);
  MPI_Type_create_subarray(ndims, sizes, subsizes, starts,   MPI_ORDER_C,
			   etype, &filetype);

  MPI_Type_commit(&etype);
  MPI_Type_commit(&array);
  MPI_Type_commit(&filetype);

  {
    MPI_File fh = MPI_FILE_NULL;
    MPI_Info info = MPI_INFO_NULL;     /* MUST BE SUPPLIED SOMEHOW */
    MPI_Offset disp = 0;

    int count = 1;

    MPI_File_open(comm, filename, MPI_MODE_WRONLY+MPI_MODE_CREATE, info, &fh);
    MPI_File_set_view(fh, disp, etype, filetype, "native", info);
    MPI_File_write_all(fh, buf->buf, count, array, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  MPI_Type_free(&filetype);
  MPI_Type_free(&array);
  MPI_Type_free(&etype);

  return 0;
}

/*****************************************************************************
 *
 *  io_aggr_mpio_read
 *
 *  SAME except write_all is read_all and mode!
 *
 *****************************************************************************/

int io_aggr_mpio_read(pe_t * pe, cs_t * cs, const char * filename,
		      io_aggr_buf_t * buf) {
  assert(pe);
  assert(cs);
  assert(filename);
  assert(buf);

  int ndims = 3;            /* nz may be 1; ie., 2-dimensional system */
  int sizes[3] = {0};       /* ie., ntotal */
  int subsizes[3] = {0};    /* ie., nlocal */
  int starts[3] = {0};      /* ie., local offset */
  int zero[3] = {0};

  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Datatype etype = MPI_DATATYPE_NULL;
  MPI_Datatype array = MPI_DATATYPE_NULL;
  MPI_Datatype filetype = MPI_DATATYPE_NULL;

  cs_cart_comm(cs, &comm);
  cs_ntotal(cs, sizes);
  cs_nlocal(cs, subsizes);
  cs_nlocal_offset(cs, starts);

  /* Element type (multiple of MPI_CHAR), and file type */

  MPI_Type_contiguous(buf->szelement, MPI_CHAR, &etype);
  MPI_Type_create_subarray(ndims, subsizes, subsizes, zero, MPI_ORDER_C,
			   etype, &array);
  MPI_Type_create_subarray(ndims, sizes, subsizes, starts,  MPI_ORDER_C,
			   etype, &filetype);

  MPI_Type_commit(&etype);
  MPI_Type_commit(&array);
  MPI_Type_commit(&filetype);

  {
    MPI_File fh = MPI_FILE_NULL;
    MPI_Info info = MPI_INFO_NULL;     /* MUST BE SUPPLIED SOMEHOW */
    MPI_Offset disp = 0;

    int count = 1;

    MPI_File_open(comm, filename, MPI_MODE_RDONLY, info, &fh);
    MPI_File_set_view(fh, disp, etype, filetype, "native", info);
    MPI_File_read_all(fh, buf->buf, count, array, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  }

  MPI_Type_free(&filetype);
  MPI_Type_free(&array);
  MPI_Type_free(&etype);

  return 0;
}
