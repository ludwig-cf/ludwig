/*****************************************************************************
 *
 *  test_io_impl_mpio.c
 *
 *  The MPI / IO implementation (with a mock aggregator).
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <inttypes.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "io_impl_mpio.h"

int test_io_impl_mpio_create(cs_t * cs, const io_metadata_t * metadata);
int test_io_impl_mpio_initialise(cs_t * cs, const io_metadata_t * metadata);

int test_io_impl_mpio_write(cs_t * cs, const io_metadata_t * metadata,
			    const char * filename);
int test_io_impl_mpio_read(cs_t * cs, const io_metadata_t * metadata,
			   const char * filename);
int test_io_impl_mpio_write_begin(io_impl_mpio_t ** io,
				  const io_metadata_t * meta,
				  const char * filename);
int test_io_impl_mpio_write_end(io_impl_mpio_t ** io);

static int test_buf_pack_asc(cs_t * cs, io_aggregator_t * buf);
static int test_buf_pack_bin(cs_t * cs, io_aggregator_t * buf);
static int test_buf_unpack_asc(cs_t * cs, const io_aggregator_t * buf);
static int test_buf_unpack_bin(cs_t * cs, const io_aggregator_t * buf);

/*****************************************************************************
 *
 *  test_io_impl_mpio_suite
 *
 *****************************************************************************/

int test_io_impl_mpio_suite(void) {

  /* I want a smallish size, but allow for a meaningful decomposition */
  int ntotal[3] = {16, 8, 4};

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  /* Describe test_buf_pack_asc() and test_buf_pack_bin() */
  io_mode_enum_t mode = IO_MODE_MPIIO;
  io_element_t element_asc = {.datatype = MPI_CHAR,
                              .datasize = sizeof(char),
                              .count    = 28,
                              .endian   = io_endianness()};
  io_element_t element_bin = {.datatype = MPI_INT64_T,
                              .datasize = sizeof(int64_t),
			      .count    = 1,
			      .endian   = io_endianness()};

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  /* ASCII: write then read */
  {
    io_options_t opts = io_options_with_format(mode, IO_RECORD_ASCII);
    io_metadata_t metadata = {0};
    const char * filename = "io-impl-mpio-asc.dat";
    const char * afilename = "io-impl-mpio-async-asc.dat";

    io_metadata_create(cs, &opts, &element_asc, &metadata);

    test_io_impl_mpio_create(cs, &metadata);
    test_io_impl_mpio_initialise(cs, &metadata);
    test_io_impl_mpio_write(cs, &metadata, filename);
    test_io_impl_mpio_read(cs, &metadata, filename);

    /* Asynchronous version ... just write ... */
    /* Something of a smoke test at the moment ... */
    {
      io_impl_mpio_t * io = NULL;
      test_io_impl_mpio_write_begin(&io, &metadata, afilename);
      test_io_impl_mpio_write_end(&io);
    }

    io_metadata_free(&metadata);

    MPI_Barrier(MPI_COMM_WORLD);
    if (pe_mpi_rank(pe) == 0) remove(filename);
    if (pe_mpi_rank(pe) == 0) remove(afilename);
  }

  /* Binary: write then read */
  {
    io_options_t opts = io_options_with_format(mode, IO_RECORD_BINARY);
    io_metadata_t metadata = {0};
    const char * filename = "io-impl-mpio-bin.dat";

    io_metadata_create(cs, &opts, &element_bin, &metadata);

    test_io_impl_mpio_create(cs, &metadata);
    test_io_impl_mpio_initialise(cs, &metadata);
    test_io_impl_mpio_write(cs, &metadata, filename);
    test_io_impl_mpio_read(cs, &metadata, filename);

    io_metadata_free(&metadata);

    MPI_Barrier(MPI_COMM_WORLD);
    if (pe_mpi_rank(pe) == 0) remove(filename);
  }

  /* Multiple file iogrid = {2, 1, 1} */

  if (pe_mpi_size(pe) > 1) {

    io_record_format_enum_t ior = IO_RECORD_ASCII;
    int iosize[3] = {2, 1, 1};
    io_options_t opts = io_options_with_iogrid(mode, ior, iosize);
    io_metadata_t metadata = {0};

    const char * filestub = "io-impl-mpio";
    char filename[BUFSIZ] = {0};

    io_metadata_create(cs, &opts, &element_asc, &metadata);
    io_subfile_name(&metadata.subfile, filestub, 0, filename, BUFSIZ);

    test_io_impl_mpio_create(cs, &metadata);
    test_io_impl_mpio_initialise(cs, &metadata);
    test_io_impl_mpio_write(cs, &metadata, filename);
    test_io_impl_mpio_read(cs, &metadata, filename);

    MPI_Barrier(metadata.comm);
    {
      /* Clean up */
      int rank = -1;
      MPI_Comm_rank(metadata.comm, &rank);
      if (rank == 0) remove(filename);
    }

    io_metadata_free(&metadata);
  }

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_create
 *
 *****************************************************************************/

int test_io_impl_mpio_create(cs_t * cs, const io_metadata_t * meta) {

  int ifail = 0;
  io_impl_mpio_t * io = NULL;

  ifail = io_impl_mpio_create(meta, &io);
  assert(ifail == 0);
  assert(io);

  assert(io->super.impl);
  assert(io->super.aggr);
  assert(io->super.aggr->buf);

  /* Just check we have retained the pointers.
   * Substantial checks are covered in initialise() test */

  assert(io->metadata   == meta);

  io_impl_mpio_free(&io);
  assert(io == NULL);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_initialise
 *
 *****************************************************************************/

int test_io_impl_mpio_initialise(cs_t * cs, const io_metadata_t * meta) {

  int ifail = 0;
  io_impl_mpio_t io = {0};

  assert(cs);
  assert(meta);

  io_impl_mpio_initialise(meta, &io);

  assert(io.super.impl);
  assert(io.super.aggr);
  assert(io.super.aggr->buf);

  assert(io.metadata   == meta);
  assert(io.fh         == MPI_FILE_NULL);

  /* Portability may be questioned if padding is present... */
  {
    /* Element size (bytes) */
    size_t esz = meta->element.datasize*meta->element.count;
    int mpisz = -1;
    MPI_Type_size(io.element, &mpisz);
    if ((size_t) mpisz != esz) ifail = -1;
    assert(ifail == 0);
  }

  {
    /* Local size of array (bytes) */
    int mpisz = -1;
    int lsize = cs_limits_size(meta->limits);
    size_t asz = lsize*meta->element.datasize*meta->element.count;
    MPI_Type_size(io.array, &mpisz);
    if ((size_t) mpisz != asz) ifail = -2;
    assert(ifail == 0);
  }

  {
    /* Size of file data type (bytes) ... is actually the same as
     * the local size ... */
    int mpisz = -1;
    int lsize = cs_limits_size(meta->limits);
    size_t fsz = lsize*meta->element.datasize*meta->element.count;
    MPI_Type_size(io.file, &mpisz);
    if ((size_t) mpisz != fsz) ifail = -4;
    assert(ifail == 0);
  }

  io_impl_mpio_finalise(&io);

  assert(io.super.aggr == NULL);
  assert(io.metadata == NULL);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_write
 *
 *****************************************************************************/

int test_io_impl_mpio_write(cs_t * cs, const io_metadata_t * meta,
			    const char * filename) {

  io_impl_mpio_t io = {0};

  assert(cs);
  assert(meta);
  assert(filename);

  io_impl_mpio_initialise(meta, &io);

  {
    io_aggregator_t * aggr = io.super.aggr;
    if (meta->element.datatype == MPI_CHAR) test_buf_pack_asc(cs, aggr);
    if (meta->element.datatype == MPI_INT64_T) test_buf_pack_bin(cs, aggr);
  }

  io_impl_mpio_write(&io, filename);

  io_impl_mpio_finalise(&io);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_read
 *
 *****************************************************************************/

int test_io_impl_mpio_read(cs_t * cs, const io_metadata_t * meta,
			   const char * filename) {

  io_impl_mpio_t io = {0};

  assert(cs);
  assert(filename);

  /* Read and check. The unpack does the check. */

  io_impl_mpio_initialise(meta, &io);

  io_impl_mpio_read(&io, filename);

  {
    io_aggregator_t * aggr = io.super.aggr;
    if (meta->element.datatype == MPI_CHAR) test_buf_unpack_asc(cs, aggr);
    if (meta->element.datatype == MPI_INT64_T) test_buf_unpack_bin(cs, aggr);
  }

  io_impl_mpio_finalise(&io);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_write_begin
 *
 *****************************************************************************/

int test_io_impl_mpio_write_begin(io_impl_mpio_t ** io,
				  const io_metadata_t * meta,
				  const char * filename) {
  assert(io);
  assert(meta);

  io_impl_mpio_create(meta, io);

  io_impl_mpio_write_begin(*io, filename);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_impl_mpio_write_end
 *
 *****************************************************************************/

int test_io_impl_mpio_write_end(io_impl_mpio_t ** io) {

  assert(io);

  io_impl_mpio_write_end(*io);
  io_impl_mpio_free(io);

  assert(*io == NULL);

  return 0;
}

/*****************************************************************************
 *
 *  test_unique_value
 *
 *****************************************************************************/

static int64_t test_unique_value(cs_t * cs, int ic, int jc, int kc) {

  int64_t ival = -1;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);

  {
    int ix = offset[X] + ic - 1;
    int iy = offset[Y] + jc - 1;
    int iz = offset[Z] + kc - 1;

    ival = ntotal[Z]*ntotal[Y]*ix + ntotal[Z]*iy + iz;
  }

  return ival;
}

/*****************************************************************************
 *
 *  test_buf_pack_asc
 *
 *****************************************************************************/

static int test_buf_pack_asc(cs_t * cs, io_aggregator_t * buf) {

  int ifail = 0;

  int ib = 0;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};

  assert(cs);
  assert(buf);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	{
	  /* Add 1 <= label <= ntotal for each dimension */
	  int ix = offset[X] + ic;
	  int iy = offset[Y] + jc;
	  int iz = offset[Z] + kc;
	  char cline[BUFSIZ] = {0};
	  size_t nc = 0; /* int returned but need to compare to size_t */
	  nc = sprintf(cline, "%4d %4d %4d %12" PRId64 "\n", ix, iy, iz, ival);
	  assert(nc == buf->szelement);
	  if (nc != buf->szelement) ifail += 1;
	  memcpy(buf->buf + ib*buf->szelement, cline, buf->szelement);
	}
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return ifail;
}

/*****************************************************************************
 *
 *  test_buf_unpack_asc
 *
 *****************************************************************************/

static int test_buf_unpack_asc(cs_t * cs, const io_aggregator_t * buf) {

  int ifail = 0;

  int ib = 0;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};

  assert(cs);
  assert(buf->buf);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	{
	  /* Add 1 <= label <= ntotal for each dimension */
	  int ix = offset[X] + ic;
	  int iy = offset[Y] + jc;
	  int iz = offset[Z] + kc;
	  int ixread = -1;
	  int iyread = -1;
	  int izread = -1;
	  int64_t ivalread = -1;
	  /* Note int64_t requires portable format */
	  int nc = sscanf(buf->buf + ib*buf->szelement, "%4d %4d %4d %" SCNd64,
			  &ixread, &iyread, &izread, &ivalread);

	  assert(nc == 4);
	  assert(ixread == ix);
	  assert(iyread == iy);
	  assert(izread == iz);
	  assert(ivalread == ival);
	  if (nc != 4) ifail += 1;
	  if (iz != izread) ifail += 1;
	  if (iy != iyread) ifail += 1;
	  if (ix != ixread) ifail =+ 1;
	  if (ivalread != ival) ifail += 1;
	}
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return ifail;
}


/*****************************************************************************
 *
 *  test_buf_pack_bin
 *
 *****************************************************************************/

static int test_buf_pack_bin(cs_t * cs, io_aggregator_t * buf) {

  assert(cs);
  assert(buf->buf);

  int ib = 0;
  int nlocal[3] = {0};

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	memcpy(buf->buf + ib*sizeof(int64_t), &ival, sizeof(int64_t));
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  test_buf_unpack_bin
 *
 *****************************************************************************/

static int test_buf_unpack_bin(cs_t * cs, const io_aggregator_t * buf) {

  int ifail = 0;

  int ib = 0;
  int nlocal[3] = {0};

  assert(cs);
  assert(buf->buf);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	int64_t iread = -1;
	memcpy(&iread, buf->buf + ib*sizeof(int64_t), sizeof(int64_t));
	assert(ival == iread);
	if (ival != iread) ifail += 1;
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);
  
  return ifail;
}
