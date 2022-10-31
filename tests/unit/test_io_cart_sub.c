/*****************************************************************************
 *
 *  test_io_cart_sub.c
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

#include "pe.h"
#include "io_cart_sub.h"

int test_io_cart_sub_create(pe_t * pe);
int test_io_cart_sub_util_pass(pe_t * pe, int ntotal[3], int iogrid[3]);

/*****************************************************************************
 *
 *  test_io_cart_sub_suite
 *
 *****************************************************************************/

int test_io_cart_sub_suite(void) {

  int sz = -1;
  pe_t * pe = NULL;

  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* "Serial": only one case */

  test_io_cart_sub_create(pe);

  /* Small: MPI_Comm_size() <= 8 */

  if (sz >= 2) {
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 1, 1};
      test_io_cart_sub_util_pass(pe, ntotal, iogrid);
    }
  }

  if (sz >= 4) {
    /* We expect MPI_Dims_create() to be the decomposition in cs_t */
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 2, 1};
      test_io_cart_sub_util_pass(pe, ntotal, iogrid);
    }
  }
  if (sz >= 8) {
    {
      int ntotal[3] = {8, 8, 8};
      int iogrid[3] = {2, 2, 2};
      test_io_cart_sub_util_pass(pe, ntotal, iogrid);
    }
  }

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_cart_sub_create
 *
 *  The simple case of one file: iogrid = {1,1,1}
 *
 *****************************************************************************/

int test_io_cart_sub_create(pe_t * pe) {

  cs_t * cs = NULL;

  assert(pe);

  /* Default system */
  cs_create(pe, &cs);
  cs_init(cs);

  {
    int iogrid[3] = {1, 1, 1};
    io_cart_sub_t iosub = {0};

    io_cart_sub_create(cs, iogrid, &iosub);

    {
      MPI_Comm comm = MPI_COMM_NULL;
      int myresult = MPI_UNEQUAL;

      cs_cart_comm(cs, &comm);
      MPI_Comm_compare(comm, iosub.parent, &myresult);
      assert(myresult == MPI_IDENT);

      MPI_Comm_compare(comm, iosub.comm, &myresult);
      assert(myresult == MPI_CONGRUENT);
    }

    assert(iosub.size[X]   == iogrid[X]);
    assert(iosub.size[Y]   == iogrid[Y]);
    assert(iosub.size[Z]   == iogrid[Z]);
    assert(iosub.coords[X] == 0);
    assert(iosub.coords[Y] == 0);
    assert(iosub.coords[Z] == 0);

    {
      int ntotal[3] = {0};
      cs_ntotal(cs, ntotal);
      assert(iosub.ntotal[X] == ntotal[X]);
      assert(iosub.ntotal[Y] == ntotal[Y]);
      assert(iosub.ntotal[Z] == ntotal[Z]);
      assert(iosub.nlocal[X] == ntotal[X]);  /* All one file */
      assert(iosub.nlocal[Y] == ntotal[Y]);
      assert(iosub.nlocal[Z] == ntotal[Z]);
      assert(iosub.offset[X] == 0);
      assert(iosub.offset[Y] == 0);
      assert(iosub.offset[Z] == 0);
    }

    assert(iosub.nfile == 1);
    assert(iosub.index == 0);

    io_cart_sub_free(&iosub);
    assert(iosub.comm == MPI_COMM_NULL);
  }

  {
    /* Here's one that must fail. */
    int iogrid[3] = {INT_MAX, 1, 1};
    io_cart_sub_t iosub = {0};

    assert(io_cart_sub_create(cs, iogrid, &iosub) != 0);

  }
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_cart_sub_util_pass
 *
 *****************************************************************************/

int test_io_cart_sub_util_pass(pe_t * pe, int ntotal[3], int iogrid[3]) {

  cs_t * cs = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  {
    int ifail = 0;
    io_cart_sub_t iosub = {0};

    ifail = io_cart_sub_create(cs, iogrid, &iosub);
    assert(ifail == 0);

    {
      /* Communicators must be different */
      MPI_Comm comm = MPI_COMM_NULL;
      int myresult = MPI_UNEQUAL;

      cs_cart_comm(cs, &comm);
      MPI_Comm_compare(comm, iosub.parent, &myresult);
      assert(myresult == MPI_IDENT);

      MPI_Comm_compare(comm, iosub.comm, &myresult);
      assert(myresult == MPI_UNEQUAL);
    }

    /* size == iogrid */
    assert(iosub.size[X] == iogrid[X]);
    assert(iosub.size[Y] == iogrid[Y]);
    assert(iosub.size[Z] == iogrid[Z]);

    /* One should be able to reconstruct the coords from
     * the index and the offset */

    /* ntotal[] is fixed */
    assert(iosub.ntotal[X] == ntotal[X]);
    assert(iosub.ntotal[Y] == ntotal[Y]);
    assert(iosub.ntotal[Z] == ntotal[Z]);
    /* PENDING NLOCAL */
    /* PENDING OFFSET */

    {
      /* Don't care about exactly what the index is ... */
      int nfile = iogrid[X]*iogrid[Y]*iogrid[Z];
      assert(iosub.nfile == nfile);
      assert(0 <= iosub.index && iosub.index < nfile);
    }

    io_cart_sub_free(&iosub);
  }

  cs_free(cs);

  return 0;
}
