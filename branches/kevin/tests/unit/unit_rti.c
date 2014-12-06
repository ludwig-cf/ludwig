/*****************************************************************************
 *
 *  unit_rti.c
 *
 *  Test the runtime interface.
 *
 *  Associated test input files are:
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "unit_control.h"

int do_test_rti1(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_rti
 *
 *****************************************************************************/

int do_ut_rti(control_t * ctrl) {

  assert(ctrl);

  do_test_rti1(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_rti1
 *
 *****************************************************************************/

int do_test_rti1(control_t * ctrl) {

  char * filename = "junk";
  int unit_rti1_input(control_t * ctrl, char * f) throws(MPIIOException);
  int unit_rti1_tests(control_t * ctrl, char * f) throws(TestFailedException);

  pe_t * pe = NULL;

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Runtine input from file\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);

  try {
    unit_rti1_input(ctrl, filename);
    unit_rti1_tests(ctrl, filename);
    remove(filename);
  }
  catch (MPIIOException) {
    /* No file :( IFAIL */
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    pe_free(&pe);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  unit_rti1_input
 *
 *****************************************************************************/

int unit_rti1_input(control_t * ctrl, char * filename) throws(MPIIOException) {

  int rank;
  e4c_mpi_t e;
  MPI_Comm comm;
  FILE * fp = NULL;

  assert(ctrl);

  control_comm(ctrl, &comm);
  MPI_Comm_rank(comm, &rank);

  e4c_mpi_init(e, comm);

  try {
    /* Only root writes the file as we only want one. */
    if (rank == 0) {
      fp = fopen(filename, "w");
      if (fp == NULL) throw(IOException, strerror(errno));

      fprintf(fp, "# Test file\n");
      fprintf(fp, "int_scalar 999\n");
      fprintf(fp, "double_scalar 3.33\n");
      fprintf(fp, "int_vector -1_-2_3\n");
      fprintf(fp, "double_vector -1.0_-2.0_+3.0\n");
      fprintf(fp, "int_multiple_space       -2\n");
      fprintf(fp, "double_tab	-2.0\n");
      fprintf(fp, "temp +1\n");
      fprintf(fp, "temperature -1\n");
      fprintf(fp, "temper 0\n");
      fprintf(fp, "string_parameter ASCII\n");
      fprintf(fp, "input_config config.0\n");

      if (ferror(fp)) throw(IOException, strerror(errno));
      fclose(fp);
    }
  }
  catch (IOException) {
    e4c_mpi_exception(e);
  }

  if (e4c_mpi_allreduce(e)) throw(MPIIOException, "");

  return 0;
}

/*****************************************************************************
 *
 *  unit_rti1_tests
 *
 *****************************************************************************/

int unit_rti1_tests(control_t * ctrl, char * filename) throws(TestFailedException) {

  char string[BUFSIZ];
  int n;
  int ivalue;
  int ivector[3];
  double dvalue;
  double dvector[3];

  assert(ctrl);

  control_verb(ctrl, "Passing %s to rti_read_input_file()\n", filename);
  RUN_read_input_file(filename); /* Can be fatal */

  /* Start the tests. */

  n = 0;
  n = RUN_get_active_keys();

  control_verb(ctrl, "Number of keys available is %d (11)\n", n);
  control_macro_test(ctrl, n == 11);

  n = 0;
  n = RUN_get_int_parameter("int_scalar", &ivalue);

  control_verb(ctrl, "key 'int_scalar' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'int_scalar' has value %d (999)\n", ivalue);
  control_macro_test(ctrl, ivalue = 999);

  n = 0;
  n = RUN_get_double_parameter("double_scalar", &dvalue);

  control_verb(ctrl, "key 'double_scalar' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'double_scalar' has value %6.3f (3.33)\n", dvalue);
  control_macro_test_dbl_eq(ctrl, dvalue, 3.33, DBL_EPSILON);

  n = 0;
  n = RUN_get_int_parameter("temperature", &ivalue);

  control_verb(ctrl, "key 'temperature' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'temperature' is: %d (-1)\n", ivalue);
  control_macro_test(ctrl, ivalue == -1);

  n = 0;
  n = RUN_get_int_parameter("temp", &ivalue);

  control_verb(ctrl, "key 'temp' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'temp' is: %d (+1)\n", ivalue);
  control_macro_test(ctrl, ivalue == 1);

  n = 0;
  n = RUN_get_int_parameter("temper", &ivalue);

  control_verb(ctrl, "key 'temper' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'temper' is: %d (0)\n", ivalue);
  control_macro_test(ctrl, ivalue == 0);

  n = 0;
  n = RUN_get_int_parameter_vector("int_vector", ivector);

  control_verb(ctrl, "key 'int_vector' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'int_vector' [x]: %d (-1)\n", ivector[0]);
  control_macro_test(ctrl, ivector[0] == -1);
  control_verb(ctrl, "key 'int_vector' [y]: %d (-2)\n", ivector[1]);
  control_macro_test(ctrl, ivector[1] == -2);
  control_verb(ctrl, "key 'int_vector' [z]: %d (+3)\n", ivector[2]);
  control_macro_test(ctrl, ivector[2] == +3);

  n = 0;
  n = RUN_get_double_parameter_vector("double_vector", dvector);

  control_verb(ctrl, "key 'double_vector' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'double_vector' [x]: %6.3f (-1.0)\n", dvector[0]);
  control_macro_test_dbl_eq(ctrl, -1.0, dvector[0], DBL_EPSILON);
  control_verb(ctrl, "key 'double_vector' [y]: %6.3f (-2.0)\n", dvector[1]);
  control_macro_test_dbl_eq(ctrl, -2.0, dvector[1], DBL_EPSILON);
  control_verb(ctrl, "key 'double_vector' [z]: %6.3f (+3.0)\n", dvector[2]);
  control_macro_test_dbl_eq(ctrl, +3.0, dvector[2], DBL_EPSILON);

  n = RUN_get_int_parameter("int_dummy", &ivalue);
  control_verb(ctrl, "key 'int_dummy' does not exist: %d\n", n);
  control_macro_test(ctrl, n == 0);

  n = RUN_get_double_parameter("double_dummy", &dvalue);
  control_verb(ctrl, "key 'double_dummy' does not exist: %d\n", n);
  control_macro_test(ctrl, n == 0);

  /* Parameters specified in odd syntax */

  n = 0;
  n = RUN_get_int_parameter("int_multiple_space", &ivalue);

  control_verb(ctrl, "key 'int_multiple_space' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'int_multiple_space' is: %d (-2)\n", ivalue);
  control_macro_test(ctrl, ivalue == -2);

  n = 0;
  n = RUN_get_double_parameter("double_tab", &dvalue);

  control_verb(ctrl, "key 'double_tab' available: %d", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'double_tab' is: %6.3f (-2.0)\n", dvalue);
  control_macro_test_dbl_eq(ctrl, dvalue, -2.0, DBL_EPSILON);

  /* String parameters */

  n = 0;
  n = RUN_get_string_parameter("string_parameter", string, BUFSIZ);

  control_verb(ctrl, "key 'string_parameter' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'string_parameter' is: %s ('ASCII')\n", string);
  control_macro_test(ctrl, strcmp(string, "ASCII") == 0);

  n = 0;
  n = RUN_get_string_parameter("input_config", string, BUFSIZ);

  control_verb(ctrl, "key 'input_config' available: %d\n", n);
  control_macro_test(ctrl, n == 1);
  control_verb(ctrl, "key 'input_config' is: %s ('config.0')\n", string);
  control_macro_test(ctrl, strcmp(string, "config.0") == 0);

  /* Finish. */

  n = 1;
  n = RUN_get_active_keys();

  control_verb(ctrl, "Keys unchecked: %d (0)\n", n);
  control_macro_test(ctrl, n == 0);

  return 0;
}

