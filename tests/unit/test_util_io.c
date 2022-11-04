/*****************************************************************************
 *
 *  test_util_io.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2002 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "util_io.h"

int test_util_io_string_to_mpi_datatype(void);
int test_util_io_mpi_datatype_to_string(void);

/*****************************************************************************
 *
 *  test_util_io_suite
 *
 *****************************************************************************/

int test_util_io_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_io_string_to_mpi_datatype();
  test_util_io_mpi_datatype_to_string();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_io_string_to_mpi_datatype
 *
 *****************************************************************************/

int test_util_io_string_to_mpi_datatype(void) {

  {
    MPI_Datatype dt = util_io_string_to_mpi_datatype(NULL);
    assert(dt == MPI_DATATYPE_NULL);
  }

  {
    MPI_Datatype dt = util_io_string_to_mpi_datatype("MPI_DOUBLE");
    assert(dt == MPI_DOUBLE);
  }

  {
    MPI_Datatype dt = util_io_string_to_mpi_datatype("MPI_INT");
    assert(dt == MPI_INT);
  }

  /* An unrecognised value -> MPI_DATATYPE_NULL */
  {
    MPI_Datatype dt = util_io_string_to_mpi_datatype("RUBBISH");
    assert(dt == MPI_DATATYPE_NULL);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_util_io_mpi_datatype_to_string
 *
 *****************************************************************************/

int test_util_io_mpi_datatype_to_string(void) {

  int ifail = 0;

  {
    const char * str = util_io_mpi_datatype_to_string(MPI_DATATYPE_NULL);
    ifail += strcmp(str, "MPI_DATATYPE_NULL");
    assert(ifail == 0);
  }

  {
    const char * str = util_io_mpi_datatype_to_string(MPI_DOUBLE);
    ifail += strcmp(str, "MPI_DOUBLE");
    assert(ifail == 0);
  }

  {
    const char * str = util_io_mpi_datatype_to_string(MPI_INT);
    ifail += strcmp(str, "MPI_INT");
    assert(ifail == 0);
  }

  return ifail;
}
