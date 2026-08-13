/*****************************************************************************
 *
 *  test_colloid_io_impl_mpio.c
 *
 *  File assets:
 *    colloid-io-mpio-read-ascii.dat
 *    colloid-io-mpio-read-ascii.dat
 *
 *  Actually symbolic links to the ansi files.
 *
 *  See also test_colloid_io_impl_ansi.c for comments on file assets.
 *
 *
 *  Edinburgh Soft Matter and Statisical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "colloid_io_impl_mpio.h"

int test_colloid_io_mpio_initialise(pe_t * pe);
int test_colloid_io_mpio_finalise(pe_t * pe);
int test_colloid_io_mpio_create(pe_t * pe);
int test_colloid_io_mpio_free(pe_t * pe);

int test_colloid_io_mpio_read(pe_t * pe);
int test_colloid_io_mpio_write(pe_t * pe);

/*****************************************************************************
 *
 *  test_colloid_io_impl_mpio_suite
 *
 *****************************************************************************/

int test_colloid_io_impl_mpio_suite(void) {

  int    ifail = 0;
  pe_t * pe    = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* The implementation/alignment of MPI_Comm makes the size 80 or 88 ... */
  /* ... so won't test for sizeof() */

  test_colloid_io_mpio_initialise(pe);
  test_colloid_io_mpio_finalise(pe);
  test_colloid_io_mpio_create(pe);
  test_colloid_io_mpio_free(pe);

  test_colloid_io_mpio_read(pe);
  test_colloid_io_mpio_write(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_mpio_initialise
 *
 *****************************************************************************/

int test_colloid_io_mpio_initialise(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    int               ncell[3] = {8, 8, 8};
    colloid_options_t options  = colloid_options_ncell(ncell);
    colloids_info_t * info     = NULL;
    colloid_io_mpio_t io       = {0};

    colloids_info_create(pe, cs, &options, &info);
    ifail = colloid_io_mpio_initialise(info, &io);
    assert(ifail == 0);

    /* Subfile must be single file: */
    assert(io.subfile.nfile == 1);
    assert(io.subfile.index == 0);

    /* Communicator */
    assert(io.comm != MPI_COMM_NULL);

    colloid_io_mpio_finalise(&io);
    colloids_info_free(&info);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_mpio_finalise
 *
 *****************************************************************************/

int test_colloid_io_mpio_finalise(pe_t * pe) {

  int    ifail    = 0;
  int    ncell[3] = {8, 8, 8};
  cs_t * cs       = NULL;

  colloid_options_t opts = colloid_options_ncell(ncell);
  colloids_info_t * info = NULL;
  colloid_io_mpio_t io   = {0};

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, &opts, &info);

  ifail = colloid_io_mpio_initialise(info, &io);
  colloid_io_mpio_finalise(&io);
  assert(io.info == NULL);
  assert(io.comm == MPI_COMM_NULL);

  colloids_info_free(&info);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_mpio_create
 *
 *****************************************************************************/

int test_colloid_io_mpio_create(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    int                 ncell[3] = {8, 8, 8};
    colloid_options_t   options  = colloid_options_ncell(ncell);
    colloids_info_t *   info     = NULL;
    colloid_io_mpio_t * io       = NULL;

    colloids_info_create(pe, cs, &options, &info);
    ifail = colloid_io_mpio_create(info, &io);
    assert(ifail == 0);
    assert(io->comm != MPI_COMM_NULL);

    colloid_io_mpio_free(&io);
    colloids_info_free(&info);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_impl_mpio_free
 *
 *****************************************************************************/

int test_colloid_io_mpio_free(pe_t * pe) {

  int    ifail    = 0;
  int    ncell[3] = {8, 8, 8};
  cs_t * cs       = NULL;

  colloid_options_t   opts = colloid_options_ncell(ncell);
  colloids_info_t *   info = NULL;
  colloid_io_mpio_t * io   = NULL;

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, &opts, &info);

  /* The test, such as it is ... */
  ifail = colloid_io_mpio_create(info, &io);
  colloid_io_mpio_free(&io);
  assert(ifail == 0);
  assert(io == NULL);

  colloids_info_free(&info);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_mpio_read
 *
 *****************************************************************************/

int test_colloid_io_mpio_read(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  /* Read from existing ASCII file. */
  /* Using the explicit interface, one doesn't need to set
   * options io mode. This could be an error. */
  {
    int                 ncell[3] = {3, 3, 3};
    colloid_options_t   opts     = colloid_options_ncell(ncell);
    colloids_info_t     info     = {0};
    colloid_io_mpio_t * io       = NULL;

    colloids_info_initialise(pe, cs, &opts, &info);

    /* ASCII read ... */

    ifail = colloid_io_mpio_create(&info, &io);
    assert(ifail == 0);

    ifail = colloid_io_mpio_read(io, "colloid-io-mpio-read-ascii.dat");
    assert(ifail == 0);
    assert(io->info->ntotal == 102); /* Total after read, all ranks */

    colloid_io_mpio_free(&io);
    colloids_info_finalise(&info);
  }

  /* Read from existing binary file */
  /* For the abstract type, one must set the relevant options io mode */
  {
    int               ncell[3] = {3, 3, 3};
    colloid_options_t opts     = colloid_options_ncell(ncell);
    colloids_info_t   info     = {0};

    /* Switch input mode and record format to non-default binary ... */
    opts.input.mode      = COLLOID_IO_MODE_MPIIO;
    opts.input.iorformat = IO_RECORD_BINARY;
    colloids_info_initialise(pe, cs, &opts, &info);

    /* ... and read via abstract type */
    {
      colloid_io_impl_t * io = NULL;

      ifail = colloid_io_impl_input(&info, &io);
      assert(ifail == 0);

      ifail = io->impl->read(io, "colloid-io-mpio-read-binary.dat");
      assert(ifail == 0);
      assert(info.ntotal == 49); /* Total after read, all ranks */

      io->impl->free(&io);
    }

    colloids_info_finalise(&info);
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_io_mpio_write
 *
 *****************************************************************************/

int test_colloid_io_mpio_write(pe_t * pe) {

  int    ifail = 0;
  cs_t * cs    = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  /* ASCII; read the existing file to generate data. */
  {
    int                 ncell[3] = {3, 3, 3};
    colloid_options_t   opts     = colloid_options_ncell(ncell);
    colloids_info_t     info     = {0};
    colloid_io_mpio_t * io       = NULL;

    colloids_info_initialise(pe, cs, &opts, &info);
    ifail = colloid_io_mpio_create(&info, &io);

    ifail = colloid_io_mpio_read(io, "colloid-io-mpio-read-ascii.dat");
    assert(ifail == 0);

    ifail = colloid_io_mpio_write(io, "colloid-mpio-write-ascii.dat");
    assert(ifail == 0);

    colloid_io_mpio_free(&io);
    colloids_info_finalise(&info);
  }

  /* Binary */
  {
    int               ncell[3] = {3, 3, 3};
    colloid_options_t opts     = colloid_options_ncell(ncell);
    colloids_info_t   info     = {0};

    opts.output.iorformat = IO_RECORD_BINARY;
    ifail                 = colloids_info_initialise(pe, cs, &opts, &info);

    /* Input, to generate some data ... */
    {
      colloid_io_mpio_t * input = NULL;

      ifail = colloid_io_mpio_create(&info, &input);
      ifail = colloid_io_mpio_read(input, "colloid-io-mpio-read-ascii.dat");
      colloid_io_mpio_free(&input);
      assert(ifail == 0);
    }

    /* Now output, via the abstract type ... */
    /* Need to set non-default output options ... */

    info.options.output.mode      = COLLOID_IO_MODE_MPIIO;
    info.options.output.iorformat = IO_RECORD_BINARY;

    {
      colloid_io_impl_t * output = NULL;

      ifail = colloid_io_impl_output(&info, &output);
      assert(ifail == 0);

      ifail = output->impl->write(output, "colloid-mpio-write-binary.dat");
      assert(ifail == 0);
      output->impl->free(&output);
    }

    colloids_info_finalise(&info);
  }

  cs_free(cs);

  return ifail;
}
