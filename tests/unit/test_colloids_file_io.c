/*****************************************************************************
 *
 *  test_colloids_file_io.c
 *
 *  File assets required:
 *
 *    colloids-file-io-read.dat
 *
 *  Actually a symbolic link.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "colloids_file_io.h"

int test_colloids_file_io_initialise(pe_t * pe);
int test_colloids_file_io_finalise(pe_t * pe);
int test_colloids_file_io_write(pe_t * pe);
int test_colloids_file_io_read(pe_t * pe);

/*****************************************************************************
 *
 *  test_colloids_file_io_suite
 *
 *****************************************************************************/

int test_colloids_file_io_suite(void) {

  int    ifail = 0;
  pe_t * pe    = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_colloids_file_io_initialise(pe);
  test_colloids_file_io_finalise(pe);

  test_colloids_file_io_read(pe);
  test_colloids_file_io_write(pe);

  pe_free(pe);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_file_io_initialise
 *
 *****************************************************************************/

int test_colloids_file_io_initialise(pe_t * pe) {

  int ifail = 0;

  /* Without on info object (ok as long as no i/o) */
  {
    colloids_info_t *  info = NULL;
    colloids_file_io_t io   = {0};

    ifail = colloids_file_io_initialise(info, &io);
    assert(ifail == 0);
    colloids_file_io_finalise(&io);
  }

  /* With an info object */
  {
    cs_t *            cs   = NULL;
    colloids_info_t * info = NULL;

    colloids_file_io_t io   = {0};
    colloid_options_t  opts = colloid_options_default();

    cs_create(pe, &cs);
    cs_init(cs);
    colloids_info_create(pe, cs, &opts, &info);

    ifail = colloids_file_io_initialise(info, &io);
    assert(ifail == 0);
    assert(io.info == info);
    if (io.input  == NULL) ifail = -1;
    if (io.output == NULL) ifail = -2;
    assert(ifail == 0);

    colloids_file_io_finalise(&io);
    colloids_info_free(&info);
    cs_free(cs);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_file_io_finalise
 *
 *****************************************************************************/

int test_colloids_file_io_finalise(pe_t * pe) {

  int ifail = 0;

  {
    colloids_info_t *  info = NULL;
    colloids_file_io_t io   = {0};

    ifail = colloids_file_io_initialise(info, &io);
    colloids_file_io_finalise(&io);
    assert(io.info   == NULL);
    assert(io.input  == NULL);
    assert(io.output == NULL);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_file_io_write
 *
 *****************************************************************************/

int test_colloids_file_io_write(pe_t * pe) {

  int ifail = 0;

  cs_t * cs = NULL;

  colloids_info_t *  info = NULL;
  colloid_options_t  opts = colloid_options_default();
  colloids_file_io_t fio  = {0};

  cs_create(pe, &cs);
  cs_init(cs);

  colloids_info_create(pe, cs, &opts, &info);

  /* Add a single particle */
  {
    colloid_state_t s = {
        .index = 1, .a0 = 2.3, .ah = 2.3, .r = {4.0, 5.0, 6.0}
    };

    ifail = colloids_info_add_state_local(info, &s);
    assert(ifail == 0);
    colloids_info_ntotal_set(info);
  }

  /* i/o */
  colloids_file_io_initialise(info, &fio);

  /* Bad filename */
  {
    ifail = colloids_file_io_write(&fio, "");
    /* Only root my get the open failure. */
    /* This broadcast would best be done in the io communicator... */
    MPI_Bcast(&ifail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    assert(ifail == MPI_ERR_NO_SUCH_FILE);
  }

  /* Valid file */
  {
    ifail = colloids_file_io_write(&fio, "test-colloids-file-io-write.dat");
    assert(ifail == MPI_SUCCESS);
  }

  colloids_file_io_finalise(&fio);
  colloids_info_free(&info);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_file_io_read
 *
 *****************************************************************************/

int test_colloids_file_io_read(pe_t * pe) {

  int ifail = 0;

  cs_t *             cs   = NULL;
  colloids_info_t *  info = NULL;
  colloid_options_t  opts = colloid_options_default();
  colloids_file_io_t fio  = {0};

  cs_create(pe, &cs);
  cs_init(cs);

  colloids_info_create(pe, cs, &opts, &info);
  colloids_file_io_initialise(info, &fio);

  /* Non-existent file */
  {
    ifail = colloids_file_io_read(&fio, "non-existent-file.dat");
    assert(ifail == MPI_ERR_NO_SUCH_FILE);
  }

  /* Badly formed ascii file */
  {
    ifail =
      colloids_file_io_read(&fio, "test-colloids-file-io-read-badly-formed.dat");
    assert(ifail == MPI_ERR_IO);
  }

  /* Valid file (ascii) */
  {
    ifail = colloids_file_io_read(&fio, "colloids-file-io-read.dat");
    assert(ifail == MPI_SUCCESS);
  }

  colloids_file_io_finalise(&fio);
  colloids_info_free(&info);
  cs_free(cs);

  return ifail;
}
