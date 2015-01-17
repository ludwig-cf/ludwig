/*****************************************************************************
 *
 *  unit_lang.c
 *
 *  Test basic language assumptions, portability issues.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008-2014 The University of Edinburgh 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdlib.h>

#include "unit_control.h"

int do_test_lang_assumptions(control_t * ctrl);
int do_test_lang_misc(control_t * ctrl);
int do_test_lang_rank_failure(control_t * ctrl);
int do_test_lang_comm_failure(control_t * ctrl);
int do_test_that_fails1(control_t * ctrl);
int do_test_that_fails2(control_t * ctrl);

/* C standard: enumeration constants are of type int */

typedef enum enum_i4 {ENUM_I4 = 256} enum_t;

/*****************************************************************************
 *
 *  do_ut_lang
 *
 *****************************************************************************/

int do_ut_lang(control_t * ctrl) {

  do_test_that_fails1(ctrl);
  do_test_that_fails2(ctrl);
  do_test_lang_assumptions(ctrl);
  do_test_lang_misc(ctrl);
  do_test_lang_rank_failure(ctrl);
  do_test_lang_comm_failure(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_that_fails1
 *
 *  One rank fails.
 *
 *****************************************************************************/

int do_test_that_fails1(control_t * ctrl) {
 
  int rank;
  int sz;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &comm);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &sz);

  try {
    control_macro_test(ctrl, rank < sz - 1);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_that_fails2
 *
 *  Rank 0, ... sz - 2 fail; rank sz - 1 passes for sz > 1.
 *
 *****************************************************************************/

int do_test_that_fails2(control_t * ctrl) {
 
  int rank;
  int sz;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &comm);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &sz);

  try {
    control_macro_test(ctrl, rank != 0  && rank == sz - 1);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_sizeof
 *
 *****************************************************************************/

int do_test_lang_assumptions(control_t * ctrl) {

  size_t nsz;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &comm);

  control_verb(ctrl, "Testing assumptions...\n");
  control_verb(ctrl, "This code compiled at %s on %s\n", __TIME__, __DATE__);

  /* All integers in the code should be declared 'int', which
   * we expect to be (at the least) 4 bytes. */
  /* All floating point types in the code should be double,
   * which must be 8 bytes. */

  try {

    nsz = sizeof(int);
    control_verb(ctrl, "sizeof(int) is %d bytes\n", nsz);
    control_macro_assert(ctrl, nsz == 4, MPITestFailedException);

    nsz = sizeof(long int);
    control_verb(ctrl, "sizeof(long int) is %d bytes\n", nsz);
    control_macro_assert(ctrl, nsz >= 4, MPITestFailedException);

    nsz = sizeof(float);
    control_verb(ctrl, "sizeof(float) is %d bytes\n", nsz);
    control_macro_assert(ctrl, nsz == 4, MPITestFailedException);

    nsz = sizeof(double);
    control_verb(ctrl, "sizeof(double) is %d bytes\n", nsz);
    control_macro_assert(ctrl, nsz == 8, MPITestFailedException);

    nsz = BUFSIZ;
    control_verb(ctrl, "BUFSIZ is %d)\n", nsz);
    control_macro_assert(ctrl, nsz >= 128, MPITestFailedException);

    nsz = FILENAME_MAX;
    control_verb(ctrl, "FILENAME_MAX is %d\n", nsz);
    control_macro_assert(ctrl, nsz >= 128, MPITestFailedException);

    nsz = sizeof(enum_t);
    control_verb(ctrl, "sizeof(enum_t) is %d\n", nsz);
    control_macro_assert(ctrl, nsz == sizeof(int), MPITestFailedException); 
  }
  catch (MPITestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_lang_misc
 *
 *****************************************************************************/

int do_test_lang_misc(control_t * ctrl) {

  int * p = NULL;

  control_test(ctrl, __CONTROL_INFO__);

  control_verb(ctrl, "sizeof(char) is %lu\n", sizeof(char));
  control_verb(ctrl, "sizeof(unsigned char) is %lu\n", sizeof(unsigned char));
  control_verb(ctrl, "sizeof(void *) is %lu\n", sizeof(void *));

  /* Zero sized alllocation */
  p = (int *) malloc(0);

  if (p == NULL) {
    control_verb(ctrl, "malloc(0) returns NULL pointer\n");
  }
  else {
    control_verb(ctrl, "malloc(0) returns non-NULL pointer\n");
    free(p);
  }

  /* Language */

  control_verb(ctrl, "__STDC__ = %d\n", __STDC__);

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_lang_c99
 *
 *****************************************************************************/

int do_test_lang_c99(control_t * ctrl) {

  control_test(ctrl, __CONTROL_INFO__);

  /*
  control_verb(ctrl, "__STDC__VERSION__ = %ld\n", __STDC__VERSION__);
  */

  return 0;
}

/*****************************************************************************
 *
 *  do_test_lang_rank_failure
 *
 *****************************************************************************/

int do_test_lang_rank_failure(control_t * ctrl) {

  int rank;
  double r1, r2;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &comm);

  MPI_Comm_rank(comm, &rank);

  r1 = 1.0;
  r2 = 2.0;
  if (rank == 0) r2 = 1.0;

  try {
    control_macro_test_dbl_eq(ctrl, r1, r2, DBL_EPSILON);
  }
  catch (TestFailedException) {
    /* rank 0 should not fail */
    if (rank == 0) control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_lang_comm_failure
 *
 *****************************************************************************/

int do_test_lang_comm_failure(control_t * ctrl) {

  int * ntry = NULL;
  int rank;
  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_comm(ctrl, &comm);
  MPI_Comm_rank(comm, &rank);

  try {
    /* We assume this really will succeed if required ... */
    if (rank > 0) ntry = (int *) calloc(1, sizeof(int));
    if (ntry == NULL) throw(NullPointerException, "");
    free(ntry);
  }
  catch (NullPointerException) {
    if (rank != 0) control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}
