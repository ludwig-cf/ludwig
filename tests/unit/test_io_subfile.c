/*****************************************************************************
 *
 *  test_io_subfile.c
 *
 *  This is one place where it would be useful to test MPI decompositions
 *  of significant extent in three dimensions.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford(kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords_s.h"
#include "io_subfile.h"

int test_io_subfile_default(pe_t * pe);
int test_io_subfile_create(pe_t * pe, int ntotal[3], int iogrid[3]);
int test_io_subfile_to_json(void);
int test_io_subfile_from_json(void);
int test_io_subfile_name(pe_t * pe, int iogrid[3]);
int test_io_subfile(cs_t * cs, int iogrid[3], const io_subfile_t * subfile);

/*****************************************************************************
 *
 *  test_io_subfile_suite
 *
 *****************************************************************************/

int test_io_subfile_suite(void) {

  int sz = -1;
  pe_t * pe = NULL;

  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* If the structure changes size, the tests change */
  assert(sizeof(io_subfile_t) == 60);

  test_io_subfile_default(pe);
  test_io_subfile_to_json();
  test_io_subfile_from_json();

  /* "Serial": iosize = {1, 1, 1} always works */
  {
    int ntotal[3] = {11, 13, 15};
    int iosize[3] = {1, 1, 1};
    test_io_subfile_create(pe, ntotal, iosize);
    test_io_subfile_name(pe, iosize);
  }

  /* Small: MPI_Comm_size() <= 8 */

  if (sz >= 2) {
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 1, 1};
      test_io_subfile_create(pe, ntotal, iogrid);
      test_io_subfile_name(pe, iogrid);
    }
    {
      int ntotal[3] = {15, 8, 8};
      int iosize[3] = { 2, 1, 1};
      test_io_subfile_create(pe, ntotal, iosize);
    }
  }

  if (sz >= 4) {
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 2, 1};
      test_io_subfile_create(pe, ntotal, iogrid);
    }
  }

  /* Medium sz >= 8 */

  if (sz >= 8) {
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 2, 2};
      test_io_subfile_create(pe, ntotal, iogrid);
    }
    {
      int ntotal[3] = {32, 16, 8};
      int iogrid[3] = {4, 2, 1};
      test_io_subfile_create(pe, ntotal, iogrid);
    }
  }

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_subfile_default
 *
 *****************************************************************************/

int test_io_subfile_default(pe_t * pe) {

  int ifail = 0;
  cs_t * cs = NULL;
  int ntotal[3] = {16, 18, 20};

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  {
    /* Always a {1, 1, 1} decomposition */
    io_subfile_t subfile = io_subfile_default(cs);
    int nlocal[3] = {0};

    cs_ntotal(cs, ntotal);
    cs_nlocal(cs, nlocal);

    assert(subfile.index == 0);
    assert(subfile.nfile == 1);

    assert(subfile.iosize[X] == 1);
    assert(subfile.iosize[Y] == 1);
    assert(subfile.iosize[Z] == 1);
    assert(subfile.coords[X] == 0);
    assert(subfile.coords[Y] == 0);
    assert(subfile.coords[Z] == 0);
    assert(subfile.offset[X] == 0);
    assert(subfile.offset[Y] == 0);
    assert(subfile.offset[Z] == 0);

    assert(subfile.ndims       == 3);
    assert(subfile.sizes[X]    == ntotal[X]);
    assert(subfile.sizes[Y]    == ntotal[Y]);
    assert(subfile.sizes[Z]    == ntotal[Z]);
    if (subfile.ndims != 3) ifail = -1;
  }

  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_subfile_create
 *
 *****************************************************************************/

int test_io_subfile_create(pe_t * pe, int ntotal[3], int iogrid[3]) {

  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_decomposition_set(cs, iogrid); /* Try 1 rank per file */
  cs_init(cs);

  {
    io_subfile_t subfile = {0};

    io_subfile_create(cs, iogrid, &subfile);
    test_io_subfile(cs, iogrid, &subfile);
  }

  cs_free(cs);

  return 0;
}


/*****************************************************************************
 *
 *  test_io_subfile
 *
 *  Check the io_subfile_t is consistent with cs_t and iogrid.
 *
 *****************************************************************************/

int test_io_subfile(cs_t * cs, int iogrid[3], const io_subfile_t * subfile) {

  int ifail = 0;
  MPI_Comm comm = MPI_COMM_NULL;
  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int offset[3] = {0};
  int cartsz[3] = {0};

  assert(cs);
  assert(subfile);

  /* Check */
  cs_cart_comm(cs, &comm);
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);
  cs_cartsz(cs, cartsz);

  {
    /* File and indices */
    int nfile = iogrid[X]*iogrid[Y]*iogrid[Z];

    assert(0 <= subfile->index && subfile->index < nfile);
    assert(subfile->nfile == nfile);
    if (subfile->nfile != nfile) ifail = -1;
  }

  assert(subfile->iosize[X] == iogrid[X]);
  assert(subfile->iosize[Y] == iogrid[Y]);
  assert(subfile->iosize[Z] == iogrid[Z]);

  assert(0 <= subfile->coords[X] && subfile->coords[X] < iogrid[X]);
  assert(0 <= subfile->coords[Y] && subfile->coords[Y] < iogrid[Y]);
  assert(0 <= subfile->coords[Z] && subfile->coords[Z] < iogrid[Z]);

  assert(subfile->ndims == 3);

  /* Sizes: the sum of the relevant ranks' nlocal */

  for (int ia = 0; ia < 3; ia++) {
    int perio = cartsz[ia]/iogrid[ia];
    int p0 = subfile->coords[ia]*perio;
    int sz = 0;
    for (int ib = p0; ib < p0 + perio; ib++) {
      sz += cs->listnlocal[ia][ib];
    }
    assert(sz == subfile->sizes[ia]);
  }

  /* (File) Offset */
  for (int ia = 0; ia < 3; ia++) {
    int offsetrank = (cartsz[ia]/iogrid[ia])*subfile->coords[ia];
    assert(subfile->offset[ia] == cs->listnoffset[ia][offsetrank]);
    if (subfile->offset[ia] != cs->listnoffset[ia][offsetrank]) ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_subfile_to_json
 *
 *****************************************************************************/

int test_io_subfile_to_json(void) {

  int ifail = 0;

  /* I'm not sure this is a particularly consistent example, but ... */
  io_subfile_t subfile = {.nfile = 8,
                          .index = 1,
                          .iosize = {1, 2, 4},
                          .coords = {0, 1, 3},
			  .ndims  = 3,
			  .sizes  = {16, 32, 64},
			  .offset = {8, 16, 21}};
  cJSON * json = NULL;
  ifail = io_subfile_to_json(&subfile, &json);
  assert(ifail == 0);
  assert(json != NULL);

  {
    /* We're going to do a slightly circular test ... */
    io_subfile_t subtest = {0};
    ifail = io_subfile_from_json(json, &subtest);
    assert(ifail == 0);
    assert(subtest.nfile     == subfile.nfile);
    assert(subtest.index     == subfile.index);
    assert(subtest.iosize[X] == subfile.iosize[X]);
    assert(subtest.iosize[Y] == subfile.iosize[Y]);
    assert(subtest.iosize[Z] == subfile.iosize[Z]);
    assert(subtest.coords[X] == subfile.coords[X]);
    assert(subtest.coords[Y] == subfile.coords[Y]);
    assert(subtest.coords[Z] == subfile.coords[Z]);
    assert(subtest.ndims     == 3);
    assert(subtest.sizes[X]  == subfile.sizes[X]);
    assert(subtest.sizes[Y]  == subfile.sizes[Y]);
    assert(subtest.sizes[Z]  == subfile.sizes[Z]);
    assert(subtest.offset[X] == subfile.offset[X]);
    assert(subtest.offset[Y] == subfile.offset[Y]);
    assert(subtest.offset[Z] == subfile.offset[Z]);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_subfile_from_json
 *
 *****************************************************************************/

int test_io_subfile_from_json(void) {

  int ifail = 0;
  const char * jstr = "{\"Number of files\": 8,"
                      " \"File index\": 1,"
                      " \"Topology\": [1, 2, 4],"
                      " \"Coordinate\": [0, 1, 3],"
                      " \"Data ndims\": 3,"
                      " \"File size (sites)\": [16, 32, 64],"
                      " \"File offset (sites)\": [7,15,31] }";

  cJSON * json = cJSON_Parse(jstr);

  assert(json != NULL);

  {
    /* Check the result is correct. */
    io_subfile_t subfile = {0};
    ifail = io_subfile_from_json(json, &subfile);
    assert(ifail == 0);
    assert(subfile.nfile == 8);
    assert(subfile.index == 1);
    assert(subfile.iosize[X] == 1);
    assert(subfile.iosize[Y] == 2);
    assert(subfile.iosize[Z] == 4);
    assert(subfile.coords[X] == 0);
    assert(subfile.coords[Y] == 1);
    assert(subfile.coords[Z] == 3);

    assert(subfile.ndims == 3);
    assert(subfile.sizes[X] == 16);
    assert(subfile.sizes[Y] == 32);
    assert(subfile.sizes[Z] == 64);
    assert(subfile.offset[X] == 7);
    assert(subfile.offset[Y] == 15);
    assert(subfile.offset[Z] == 31);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_subfile_name
 *
 *****************************************************************************/

int test_io_subfile_name(pe_t * pe, int iogrid[3]) {

  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    io_subfile_t subfile = io_subfile_default(cs);
    int iteration = 0;
    char filename[BUFSIZ] = {0};

    io_subfile_create(cs, iogrid, &subfile);
    io_subfile_name(&subfile, "stub", iteration, filename, BUFSIZ);
    assert(strncmp("stub-", filename, 5) == 0);
  }

  cs_free(cs);

  return 0;
}
