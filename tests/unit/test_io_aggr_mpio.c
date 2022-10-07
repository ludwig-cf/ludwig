/*****************************************************************************
 *
 *  test_io_aggr_mpio.c
 *
 *  The MPI / IO aggregtor (with a mock aggregator).
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
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "io_aggr.h"
#include "io_aggr_buf_mpio.h"

int test_io_aggr_mpio_write(pe_t * pe, cs_t * cs, io_aggr_t aggr,
			    const char * filename);
int test_io_aggr_mpio_read(pe_t * pe, cs_t * cs, io_aggr_t aggr,
			   const char * filename);

int test_io_aggr_buf_pack_asc(cs_t * cs, io_aggr_buf_t buf);
int test_io_aggr_buf_pack_bin(cs_t * cs, io_aggr_buf_t buf);
int test_io_aggr_buf_unpack_asc(cs_t * cs, io_aggr_buf_t buf);
int test_io_aggr_buf_unpack_bin(cs_t * cs, io_aggr_buf_t buf);

/*****************************************************************************
 *
 *  test_io_aggr_mpio_suite
 *
 *****************************************************************************/

int test_io_aggr_mpio_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  cs_create(pe, &cs);

  {
    /* I want a smallish size, but allow for a meaningful decomposition */
    int ntotal[3] = {16, 8, 4};
    cs_ntotal_set(cs, ntotal);
  }

  cs_init(cs);

  /* ASCII: write then read */
  {
    io_aggr_t aggr = {.etype = MPI_CHAR, .esize = 28*sizeof(char)};
    const char * filename = "io-aggr-mpio-asc.dat";

    test_io_aggr_mpio_write(pe, cs, aggr, filename);
    test_io_aggr_mpio_read(pe, cs, aggr, filename);

    MPI_Barrier(MPI_COMM_WORLD);
    if (pe_mpi_rank(pe) == 0) remove(filename);
  }

  /* Binary: write thne read */
  {
    io_aggr_t aggr = {.etype = MPI_INT64_T, .esize = sizeof(int64_t)};
    const char * filename = "io-aggr-mpio-bin.dat";

    test_io_aggr_mpio_write(pe, cs, aggr, filename);
    test_io_aggr_mpio_read(pe, cs, aggr, filename);

    MPI_Barrier(MPI_COMM_WORLD);
    if (pe_mpi_rank(pe) == 0) remove(filename);
  }
  
  pe_info(pe, "PASS      ./unit/test_io_aggr_mpio\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_aggr_mpio_write
 *
 *****************************************************************************/

int test_io_aggr_mpio_write(pe_t * pe, cs_t * cs, io_aggr_t aggr,
			    const char * filename) {

  assert(pe);
  assert(cs);
  assert(filename);

  int nlocal[3] = {0};
  cs_nlocal(cs, nlocal);

  /* Aggregator buffer */

  {
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    io_aggr_buf_t buf = {0};

    io_aggr_buf_create(aggr.esize, lim, &buf);

    if (aggr.etype == MPI_CHAR) test_io_aggr_buf_pack_asc(cs, buf);
    if (aggr.etype == MPI_INT64_T) test_io_aggr_buf_pack_bin(cs, buf);

    io_aggr_mpio_write(pe, cs, filename, &buf);

    io_aggr_buf_free(&buf);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_io_aggr_mpio_read
 *
 *****************************************************************************/

int test_io_aggr_mpio_read(pe_t * pe, cs_t * cs, io_aggr_t aggr,
			   const char * filename) {

  assert(pe);
  assert(cs);
  assert(filename);

  int nlocal[3] = {0};
  cs_nlocal(cs, nlocal);

  {
    /* Read and check */
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    io_aggr_buf_t buf = {0};

    io_aggr_buf_create(aggr.esize, lim ,&buf);

    io_aggr_mpio_read(pe, cs, filename, &buf);

    if (aggr.etype == MPI_CHAR) test_io_aggr_buf_unpack_asc(cs, buf);
    if (aggr.etype == MPI_INT64_T) test_io_aggr_buf_unpack_bin(cs, buf);

    io_aggr_buf_free(&buf);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_unique_value
 *
 *****************************************************************************/

int64_t test_unique_value(cs_t * cs, int ic, int jc, int kc) {

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
 *  test_io_aggr_buf_pack_asc
 *
 *****************************************************************************/

int test_io_aggr_buf_pack_asc(cs_t * cs, io_aggr_buf_t buf) {

  assert(cs);
  assert(buf.buf);

  int ib = 0;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};

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
	  int nc = -1;
	  char cline[BUFSIZ] = {0};
	  nc = sprintf(cline, "%4d %4d %4d %12lld\n", ix, iy, iz, ival);
	  assert(nc == buf.szelement);
	  memcpy(buf.buf + ib*buf.szelement, cline, buf.szelement);
	}
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_aggr_buf_unpack_asc
 *
 *****************************************************************************/

int test_io_aggr_buf_unpack_asc(cs_t * cs, io_aggr_buf_t buf) {

  assert(cs);
  assert(buf.buf);

  int ib = 0;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};

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
	  int nc = sscanf(buf.buf + ib*buf.szelement, "%4d %4d %4d %12lld",
			  &ixread, &iyread, &izread, &ivalread);

	  assert(nc == 4);
	  assert(ixread == ix);
	  assert(iyread == iy);
	  assert(izread == iz);
	  assert(ivalread == ival);
	}
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return 0;
}


/*****************************************************************************
 *
 *  test_io_aggr_buf_pack_bin
 *
 *****************************************************************************/

int test_io_aggr_buf_pack_bin(cs_t * cs, io_aggr_buf_t buf) {

  assert(cs);
  assert(buf.buf);

  int ib = 0;
  int nlocal[3] = {0};

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	memcpy(buf.buf + ib*sizeof(int64_t), &ival, sizeof(int64_t));
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_aggr_buf_unpack_bin
 *
 *****************************************************************************/

int test_io_aggr_buf_unpack_bin(cs_t * cs, io_aggr_buf_t buf) {

  assert(cs);
  assert(buf.buf);

  int ib = 0;
  int nlocal[3] = {0};

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int64_t ival = test_unique_value(cs, ic, jc, kc);
	int64_t iread = -1;
	memcpy(&iread, buf.buf + ib*sizeof(int64_t), sizeof(int64_t));
	assert(ival == iread);
	ib += 1;
      }
    }
  }

  assert(ib == nlocal[X]*nlocal[Y]*nlocal[Z]);
  
  return 0;
}
