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

int do_test_pe_init(control_t * ctrl);
int do_test_pe_init_quiet(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_pe
 *
 *  There's not actually a great deal to test.
 *
 *****************************************************************************/

int do_ut_pe(control_t * ctrl) {

  do_test_pe_init_quiet(ctrl);
  do_test_pe_init(ctrl);
  
  return 0;
}

/*****************************************************************************
 *
 *  do_test_pe_init
 *
 *****************************************************************************/

int do_test_pe_init(control_t * ctrl) {

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "pe_init with introductory messages\n");

  pe_init();
  pe_finalise();

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_pe_init_quiet
 *
 *****************************************************************************/

int do_test_pe_init_quiet(control_t * ctrl) {

  int rank, sz;
  int icompare = MPI_IDENT;
  char subname[BUFSIZ];

  pe_t * pe = NULL;
  MPI_Comm comm;
  MPI_Comm ctrlcomm;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &ctrlcomm);
  control_verb(ctrl, "pe_init_quiet\n");

  pe_create_parent(ctrlcomm, &pe);
  pe_init_quiet();

  try {
    comm = pe_comm();
    MPI_Comm_compare(comm, ctrlcomm, &icompare);
    control_macro_test(ctrl, icompare == MPI_CONGRUENT);

    pe_subdirectory_set("testme");
    pe_subdirectory(subname);

    control_verb(ctrl, "testme/ and %s\n", subname);
    control_macro_test(ctrl, strcmp("testme/", subname) == 0);

    MPI_Comm_rank(ctrlcomm, &rank);
    MPI_Comm_size(ctrlcomm, &sz);

    control_macro_test(ctrl, rank == pe_rank());
    control_macro_test(ctrl, sz == pe_size());
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  pe_finalise();

  control_report(ctrl);

  return 0;
}
