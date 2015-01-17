/*****************************************************************************
 *
 *  ut_pe.c
 *
 *  Unit test for parallel environment
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>
#include "unit_control.h"

#include "pe.h"

int do_test_pe_create(control_t * ctrl);
int do_test_pe_mpi(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_pe
 *
 *  There's not actually a great deal to test.
 *
 *****************************************************************************/

int do_ut_pe(control_t * ctrl) {

  do_test_pe_create(ctrl);
  do_test_pe_mpi(ctrl);
  
  return 0;
}

/*****************************************************************************
 *
 *  do_test_pe_create
 *
 *****************************************************************************/

int do_test_pe_create(control_t * ctrl) {

  pe_t * pe = NULL;
  pe_t * pe_copy = NULL;
  MPI_Comm comm;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "pe_create_parent works...\n");
  control_comm(ctrl, &comm);

  try {
    pe_create_parent(comm, &pe);
    control_macro_test(ctrl, pe != NULL);

    pe_retain(pe);
    pe_copy = pe;
    pe_free(pe_copy);
    pe_free(pe);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_pe_mpi
 *
 *****************************************************************************/

int do_test_pe_mpi(control_t * ctrl) {

  int rank, sz;
  int icompare = MPI_IDENT;
  char subname[BUFSIZ];

  pe_t * pe = NULL;
  MPI_Comm comm;
  MPI_Comm ctrlcomm;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &ctrlcomm);
  control_verb(ctrl, "pe_mpi_comm etc\n");

  pe_create_parent(ctrlcomm, &pe);

  try {
    pe_mpi_comm(pe, &comm);
    MPI_Comm_compare(comm, ctrlcomm, &icompare);
    control_macro_test(ctrl, icompare == MPI_CONGRUENT);

    pe_subdirectory_set(pe, "testme");
    pe_subdirectory(pe, subname);

    control_verb(ctrl, "testme/ and %s\n", subname);
    control_macro_test(ctrl, strcmp("testme/", subname) == 0);

    MPI_Comm_rank(ctrlcomm, &rank);
    MPI_Comm_size(ctrlcomm, &sz);

    control_macro_test(ctrl, rank == pe_mpi_rank(pe));
    control_macro_test(ctrl, sz == pe_mpi_size(pe));
    control_macro_test(ctrl, rank == pe_mpi_rank(pe));
    control_macro_test(ctrl, sz == pe_mpi_size(pe));
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  pe_free(pe);

  control_report(ctrl);

  return 0;
}
