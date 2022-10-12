/*****************************************************************************
 *
 *  test_io_info_args_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Groups and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "io_info_args_rt.h"

int test_io_info_args_rt(pe_t * pe);
int test_io_info_args_rt_input(pe_t * pe);
int test_io_info_args_rt_output(pe_t * pe);
int test_io_info_args_rt_iogrid(pe_t * pe);

/*****************************************************************************
 *
 *  test_io_info_args_rt_suite
 *
 *****************************************************************************/

int test_io_info_args_rt_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_io_info_args_rt_iogrid(pe);
  /* PENDING DEBUG test_io_info_args_rt_input(pe); */
  test_io_info_args_rt(pe);

  pe_info(pe, "%10s %s\n", "PASS **", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_info_args_rt
 *
 *****************************************************************************/

int test_io_info_args_rt(pe_t * pe) {

  int ifail = 0;
  io_info_args_t args = io_info_args_default();
  rt_t * rt = NULL;

  rt_create(pe, &rt);

  io_info_args_rt(rt, RT_FATAL, "stub", IO_INFO_READ_WRITE, &args);

  rt_free(rt);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_info_args_rt_input
 *
 *****************************************************************************/

int test_io_info_args_rt_input(pe_t * pe) {

  int ifail = 0;
  rt_t * rt = NULL;

  rt_create(pe, &rt);

  /* No keys are present: default */
  {
    io_info_args_t args = io_info_args_default();

    io_info_args_rt_input(rt, RT_FATAL, "stub", &args);
    assert(args.input.mode             == io_mode_default());
    assert(args.input.iorformat        == io_record_format_default());
    assert(args.input.metadata_version == io_metadata_version_default());
    assert(args.input.report           == 0);
  }

  rt_add_key_value(rt, "default_io_mode",   "multiple");
  rt_add_key_value(rt, "default_io_format", "ascii");
  rt_add_key_value(rt, "default_io_report", "yes");

  /* Default keys only are present: check */
  {
    io_info_args_t args = io_info_args_default();

    io_info_args_rt_input(rt, RT_FATAL, "nostub", &args);
    assert(args.input.mode              == IO_MODE_MULTIPLE);
    assert(args.input.iorformat         == IO_RECORD_ASCII);
    assert(args.input.metadata_version  == IO_METADATA_MULTI_V1);
    assert(args.input.report            == 1);
  }

  /* Specific key stub: different from default keys */

  rt_add_key_value(rt, "phi_input_io_mode",   "single");
  rt_add_key_value(rt, "phi_input_io_format", "BINARY");
  rt_add_key_value(rt, "phi_input_io_report", "no");

  {
    io_info_args_t args = io_info_args_default();

    io_info_args_rt_input(rt, RT_FATAL, "phi", &args);
    assert(args.input.mode             == IO_MODE_SINGLE);
    assert(args.input.iorformat        == IO_RECORD_BINARY);
    assert(args.input.metadata_version == IO_METADATA_SINGLE_V1);
    assert(args.input.report           == 0);
  }

  rt_free(rt);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_info_args_rt_iogrid
 *
 *****************************************************************************/

int test_io_info_args_rt_iogrid(pe_t * pe) {

  int ierr = 0;
  rt_t * rt = NULL;

  rt_create(pe, &rt);
  rt_add_key_value(rt, "iogrid_right", "2_3_4");
  rt_add_key_value(rt, "iogrid_wrong", "0_0_0");

  /* No key at all. iogrid arg should be unchanged */
  {
    int iogrid[3] = {2, 3, 4};
    int ifail = io_info_args_rt_iogrid(rt, RT_FATAL, "iogrid_none", iogrid);
    assert(ifail == -1);
    assert(iogrid[0] == 2);
    assert(iogrid[1] == 3);
    assert(iogrid[2] == 4);
  }
  
  /* Right */
  {
    int iogrid[3] = {0};
    int ifail = io_info_args_rt_iogrid(rt, RT_FATAL, "iogrid_right", iogrid);
    assert(ifail == 0);
    assert(iogrid[0] == 2);
    assert(iogrid[1] == 3);
    assert(iogrid[2] == 4);
  }

  /* Wrong. Key present but invalid. RT_NONE for no fatal error. */
  {
    int grid[3] = {0};
    int ifail = io_info_args_rt_iogrid(rt, RT_NONE, "iogrid_wrong", grid);
    assert (ifail == +1);
  }

  rt_free(rt);

  return ierr;
}
