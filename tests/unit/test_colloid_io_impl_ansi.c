/*****************************************************************************
 *
 *  test_colloid_io_impl_ansi.c
 *
 *  File assets for this test:
 *    colloid-io-ansi-read-ascii.dat         via util/colloid_init -v 0.02 ...
 *    colloid-io-ansi-read-binary.dat        ... for size {64, 64, 64}.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc,ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include <errno.h>
#include <string.h>

#include "colloid_io_impl_ansi.h"

int test_colloid_io_ansi_initialise(pe_t * pe);
int test_colloid_io_ansi_create(pe_t * pe);

int test_colloid_io_ansi_read(pe_t * pe);
int test_colloid_io_ansi_write(pe_t * pe);

/*****************************************************************************
 *
 *  test_colloid_io_impl_ansi_suite
 *
 *****************************************************************************/

int test_colloid_io_impl_ansi_suite(void) {

  int    ifail = 0;
  pe_t * pe    = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* The implementation/alignment of MPI_Comm makes the size 80 or 88 ... */
  /* ... so won't test for sizeof() */

  test_colloid_io_ansi_initialise(pe);
  test_colloid_io_ansi_create(pe);

  test_colloid_io_ansi_read(pe);
  test_colloid_io_ansi_write(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_ansi_initialise
 *
 *****************************************************************************/

int test_colloid_io_ansi_initialise(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    int               ncell[3] = {8, 8, 8};
    colloid_options_t options  = colloid_options_ncell(ncell);
    colloids_info_t * info     = NULL;
    colloid_io_ansi_t io       = {0};

    colloids_info_create(pe, cs, &options, &info);
    ifail = colloid_io_ansi_initialise(info, &io);
    assert(ifail == 0);

    /* subfile exists ... */
    assert(io.subfile.nfile == 1);
    assert(io.subfile.index == 0);

    /* new communicator exists */
    assert(io.comm != MPI_COMM_NULL);
    assert(io.comm != cs->commcart);

    colloid_io_ansi_finalise(&io);
    colloids_info_free(&info);
    assert(io.comm == MPI_COMM_NULL);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_ansi_create
 *
 *****************************************************************************/

int test_colloid_io_ansi_create(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    int                 ncell[3] = {8, 8, 8};
    colloid_options_t   options  = colloid_options_ncell(ncell);
    colloids_info_t *   info     = NULL;
    colloid_io_ansi_t * io       = NULL;

    colloids_info_create(pe, cs, &options, &info);
    ifail = colloid_io_ansi_create(info, &io);
    assert(ifail == 0);
    assert(io->comm != MPI_COMM_NULL);

    colloid_io_ansi_free(&io);
    colloids_info_free(&info);
    assert(io == NULL);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_ansi_read
 *
 *****************************************************************************/

int test_colloid_io_ansi_read(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  /* Read from existing ASCII file. */
  {
    int                 ncell[3] = {3, 3, 3};
    colloid_options_t   opts     = colloid_options_ncell(ncell);
    colloids_info_t   * info     = NULL;
    colloid_io_ansi_t * io       = NULL;

    colloids_info_create(pe, cs, &opts, &info);

    /* ASCII read ... */

    ifail = colloid_io_ansi_create(info, &io);
    assert(ifail == 0);

    ifail = colloid_io_ansi_read(io, "colloid-io-ansi-read-ascii.dat");
    assert(ifail == 0);
    assert(io->info->ntotal == 102); /* Total after read, all ranks */

    colloid_io_ansi_free(&io);
    colloids_info_free(&info);
  }

  /* Read from existing binary file */
  {
    int               ncell[3] = {3, 3, 3};
    colloid_options_t opts     = colloid_options_ncell(ncell);
    colloids_info_t   * info     = NULL;

    /* Switch input record format to non-default binary ... */
    opts.input.iorformat = IO_RECORD_BINARY;
    colloids_info_create(pe, cs, &opts, &info);

    /* ... and read via abstract type */
    {
      colloid_io_impl_t * io = NULL;

      ifail = colloid_io_impl_input(info, &io);
      assert(ifail == 0);

      ifail = io->impl->read(io, "colloid-io-ansi-read-binary.dat");
      assert(ifail == 0);
      assert(info->ntotal == 49); /* Total after read, all ranks */

      io->impl->free(&io);
    }

    colloids_info_free(&info);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_ansi_write
 *
 *****************************************************************************/

int test_colloid_io_ansi_write(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  /* ASCII; read the exiting file to generate data. */
  {
    int                 ncell[3] = {3, 3, 3};
    colloid_options_t   opts     = colloid_options_ncell(ncell);
    colloids_info_t   * info     = NULL;
    colloid_io_ansi_t * io       = NULL;

    colloids_info_create(pe, cs, &opts, &info);
    ifail = colloid_io_ansi_create(info, &io);

    ifail = colloid_io_ansi_read(io, "colloid-io-ansi-read-ascii.dat");
    assert(ifail == 0);

    ifail = colloid_io_ansi_write(io, "colloids-ansi-write-ascii.dat");
    assert(ifail == 0);

    colloid_io_ansi_free(&io);
    colloids_info_free(&info);
  }

  /* Binary */
  {
    int               ncell[3] = {3, 3, 3};
    colloid_options_t opts     = colloid_options_ncell(ncell);
    colloids_info_t   * info     = NULL;

    opts.output.iorformat = IO_RECORD_BINARY;
    ifail                 = colloids_info_create(pe, cs, &opts, &info);

    /* Input, to generate some data ... */
    {
      colloid_io_ansi_t * input = NULL;

      ifail = colloid_io_ansi_create(info, &input);
      ifail = colloid_io_ansi_read(input, "colloid-io-ansi-read-ascii.dat");
      colloid_io_ansi_free(&input);
      assert(ifail == 0);
    }

    /* Now output, via the abstract type ... */
    {
      colloid_io_impl_t * output = NULL;

      ifail = colloid_io_impl_output(info, &output);
      assert(ifail == 0);

      ifail = output->impl->write(output, "colloid-ansi-write-binary.dat");
      assert(ifail == 0);
      output->impl->free(&output);
    }

    colloids_info_free(&info);
  }

  cs_free(cs);

  return ifail;
}
