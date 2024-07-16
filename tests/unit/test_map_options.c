/*****************************************************************************
 *
 *  test_map_options.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "map_options.h"

/* Also map_options are here */

int test_map_options_default(void);
int test_map_options_ndata(void);
int test_map_options_valid(void);

/*****************************************************************************
 *
 *  test_map_options_suite
 *
 *****************************************************************************/

int test_map_options_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_map_options_default();
  test_map_options_ndata();
  test_map_options_valid();

  /* Change of object size suggests tests should be updated. */
  assert(sizeof(map_options_t) == 104);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_options_default
 *
 *****************************************************************************/

int test_map_options_default(void) {

  int ifail = 0;
  map_options_t opts = map_options_default();

  assert(opts.ndata == 0);
  assert(opts.is_porous_media == 0);
  assert(strcmp(opts.filestub, "map") == 0);

  ifail = io_options_valid(&opts.iodata.input);
  if (ifail == 0) ifail = -1;

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_options_ndata
 *
 *****************************************************************************/

int test_map_options_ndata(void) {

  int ifail = 0;
  int ndata = 2;
  map_options_t opts = map_options_ndata(ndata);

  if (opts.ndata != ndata) ifail = -1;
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_options_valid
 *
 *****************************************************************************/

int test_map_options_valid(void) {

  int ifail = 0;

  {
    map_options_t opts = map_options_default();
    ifail = map_options_valid(&opts);
    assert(ifail != 0);
  }

  {
    map_options_t opts = map_options_ndata(2);
    ifail = map_options_valid(&opts);
    assert(ifail != 0);
  }

  {
    map_options_t opts = map_options_ndata(-1);
    ifail = map_options_valid(&opts);
    assert(ifail == 0);
  }

  return ifail;
}
