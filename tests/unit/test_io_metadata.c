/*****************************************************************************
 *
 *  test_io_metadata.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "io_metadata.h"

#define IO_METADATA_VERBOSE 0

int test_io_metadata_initialise(cs_t * cs);
int test_io_metadata_create(cs_t * cs);
int test_io_metadata_to_json(cs_t * cs);
int test_io_metadata_write(cs_t * cs, int keep);
int test_io_metadata_from_file(pe_t * pe);

/*****************************************************************************
 *
 *  test_io_metadata_suite
 *
 *****************************************************************************/

int test_io_metadata_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  cs_create(pe, &cs);
  cs_init(cs);

  test_io_metadata_initialise(cs);
  test_io_metadata_create(cs);
  test_io_metadata_to_json(cs);
  test_io_metadata_write(cs, 0);

  test_io_metadata_write(cs, 1);
  test_io_metadata_from_file(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_metadata_create
 *
 *****************************************************************************/

int test_io_metadata_create(cs_t * cs) {

  int ifail = 0;
  io_options_t options = io_options_default();
  io_element_t element = {0};
  io_metadata_t * meta = NULL;

  assert(cs);

  ifail = io_metadata_create(cs, &options, &element, &meta);
  assert(ifail == 0);

  assert(meta);

  io_metadata_free(&meta);
  assert(meta == NULL);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_metadata_initialise
 *
 *****************************************************************************/

int test_io_metadata_initialise(cs_t * cs) {

  int ifail = 0;
  io_metadata_t metadata = {0};
  io_element_t  element  = {0};
  io_options_t  options  = io_options_default();

  assert(cs);

  ifail = io_metadata_initialise(cs, &options, &element, &metadata);
  assert(ifail == 0); /* Bad decomposition */

  /* Really want a proper compare method for components */
  assert(metadata.options.iogrid[X] == options.iogrid[X]);
  assert(metadata.options.iogrid[Y] == options.iogrid[Y]);
  assert(metadata.options.iogrid[Z] == options.iogrid[Z]);

  assert(metadata.subfile.iosize[X] == options.iogrid[X]);
  assert(metadata.subfile.iosize[Y] == options.iogrid[Y]);
  assert(metadata.subfile.iosize[Z] == options.iogrid[Z]);

  {
    /* Check limits */
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);
    assert(metadata.limits.imin == 1);
    assert(metadata.limits.jmin == 1);
    assert(metadata.limits.kmin == 1);
    assert(metadata.limits.imax == nlocal[X]);
    assert(metadata.limits.jmax == nlocal[Y]);
    assert(metadata.limits.kmax == nlocal[Z]);
  }

  {
    /* Parent communicator should be the Carteisan communicator */
    /* and the communicator is congruent with it */
    MPI_Comm comm = MPI_COMM_NULL;
    int myresult  = MPI_UNEQUAL;

    cs_cart_comm(cs, &comm);
    MPI_Comm_compare(comm, metadata.parent, &myresult);
    assert(myresult == MPI_IDENT);

    MPI_Comm_compare(comm, metadata.comm, &myresult);
    assert(myresult == MPI_CONGRUENT);
  }

  io_metadata_finalise(&metadata);
  assert(metadata.comm == MPI_COMM_NULL);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_metadata_to_json
 *
 *****************************************************************************/

int test_io_metadata_to_json(cs_t * cs) {

  int ifail = 0;
  io_metadata_t metadata = {0};
  io_element_t  element  = {0};
  io_options_t  options  = io_options_default();

  assert(cs);

  io_metadata_initialise(cs, &options, &element, &metadata);

  {
    cJSON * json = NULL;
    ifail = io_metadata_to_json(&metadata, &json);
    assert(ifail == 0);

    if (IO_METADATA_VERBOSE) {
      char * str = cJSON_Print(json);
      printf("io_metadata: %s\n\n", str);
      free(str);
    }

    /* Check we have the requisite parts. */
    /* We assume the parts themselves are well-formed. */
    {
      cJSON * part = NULL;
      part = cJSON_GetObjectItemCaseSensitive(json, "io_options");
      if (part == NULL) ifail = -1;
      assert(ifail == 0);
      part = cJSON_GetObjectItemCaseSensitive(json, "io_element");
      if (part == NULL) ifail = -1;
      assert(ifail == 0);
      part = cJSON_GetObjectItemCaseSensitive(json, "io_subfile");
      if (part == NULL) ifail = -1;
      assert(ifail == 0);
    }

    cJSON_Delete(json);
  }

  io_metadata_finalise(&metadata);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_metadata_write
 *
 *****************************************************************************/

int test_io_metadata_write(cs_t * cs, int keep) {

  int ifail = 0;
  io_metadata_t * meta = NULL;

  io_element_t element = {.datatype = MPI_DOUBLE,
			  .datasize = sizeof(double),
                          .count    = 5,
                          .endian   = io_endianness()};
  io_options_t options = io_options_with_format(IO_MODE_MPIIO,
						IO_RECORD_BINARY);

  cJSON * header = cJSON_Parse("{\"Test\": \"via test_io_metadata.c\"}");
  assert(cs);

  io_metadata_create(cs, &options, &element, &meta);
  ifail = io_metadata_write(meta, "test-io", header);
  assert(ifail == 0);

  /* Remove at rank 0 (a test that the file exists with the correct name) */
  if (keep == 0) {
    int rank = -1;
    MPI_Comm_rank(meta->comm, &rank);
    if (rank == 0) ifail = remove("test-io-metadata.001-001");
    assert(ifail == 0);
  }

  io_metadata_free(&meta);
  cJSON_Delete(header);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_metadata_from_file
 *
 *  Experimental. Use the file generated by the above routine.
 *  Serial file operations.
 *
 *****************************************************************************/

int test_io_metadata_from_file(pe_t * pe) {

  int ifail = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  const char * filename = "test-io-metadata.001-001";
  io_metadata_t * metadata = NULL;

  pe_mpi_comm(pe, &comm);
  MPI_Barrier(comm);

  /* All ranks read at the moment ... */
  ifail = io_metadata_from_file(pe, filename, &metadata);
  assert(ifail == 0);

  /* Check one component of each part */
  assert(metadata->cs);
  assert(metadata->options.metadata_version == 3);
  assert(metadata->element.datatype         == MPI_DOUBLE);
  assert(metadata->subfile.sizes[X]         == 64);

  {
    /* Cleaning up at the moment requires.. */
    cs_t * cs = metadata->cs;
    io_metadata_free(&metadata);
    cs_free(cs);
  }

  MPI_Barrier(comm);
  {
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    ifail = remove(filename);
  }

  return ifail;
}
