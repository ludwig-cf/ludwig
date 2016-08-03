/*****************************************************************************
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
#include <math.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "tests.h"

/*****************************************************************************
 *
 *  test_rt_suite
 *
 *****************************************************************************/

int test_rt_suite(void) {

  int    n;
  int    ivalue;
  int    ivector[3];
  double dvalue;
  double dvector[3];
  char   string[256];
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* info("Testing runtime.c...\n");

     info("Checking can read the file 'test_runtime_input1'... \n"); */
  RUN_read_input_file("test_runtime_input1");
  test_assert(1);
  /* info("...input file read ok.\n");*/

  n = 0;
  /* info("Checking number of keys available is now 11... ");*/
  n = RUN_get_active_keys();
  test_assert(n == 15);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'int_scalar' is available...");*/
  n = RUN_get_int_parameter("int_scalar", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking key 'int_scalar' has correct value...");*/
  test_assert(ivalue == 999);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'double_scalar' is available...");*/
  n = RUN_get_double_parameter("double_scalar", &dvalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking key 'double_scalar' has correct value...");*/
  test_assert(fabs(dvalue - 3.33) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'temperature' is available...");*/
  n = RUN_get_int_parameter("temperature", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'temperature' is -1...");*/
  test_assert(ivalue == -1);
  /*info("yes\n");*/

  n = 0;
  /* info("Checking 'temp' is available...");*/
  n = RUN_get_int_parameter("temp", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'temp' is +1...");*/
  test_assert(ivalue == 1);
  /* info("yes\n");*/

  /* info("Checking 'temper' is 0...");*/
  n = RUN_get_int_parameter("temper", &ivalue);
  test_assert(ivalue == 0);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'int_vector' is available...");*/
  n = RUN_get_int_parameter_vector("int_vector", ivector);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking int_vector x component is -1 ...");*/
  test_assert(ivector[0] == -1);
  /*info("yes\n");*/

  /*info("Checking int_vector y component is -2 ...");*/
  test_assert(ivector[1] == -2);
  /* info("yes\n");*/

  /* info("Checking int_vector z component is +3 ...");*/
  test_assert(ivector[2] == 3);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'double_vector' is available ...");*/
  n = RUN_get_double_parameter_vector("double_vector", dvector);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking double_vector x component is -1.0 ...");*/
  test_assert(fabs(dvector[0] - -1.0) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  /* info("Checking double_vector y component is -2.0 ...");*/
  test_assert(fabs(dvector[1] - -2.0) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  /* info("Checking double_vector z component is +3.0 ...");*/
  test_assert(fabs(dvector[2] - 3.0) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  n = 1;
  /* info("Checking 'int_dummy' does not exist ...");*/
  n = RUN_get_int_parameter("int_dummy", &ivalue);
  test_assert(n == 0);
  /* info("ok\n");*/

  /* info("Checking 'double_dummy' does not exist ...");*/
  n = RUN_get_double_parameter("double_dummy", &dvalue);
  test_assert(n == 0);
  /* info("ok\n");*/

  /* Parameters specified in odd syntax */

  n = 0;
  /* info("Checking 'int_multiple_space' is available...");*/
  n = RUN_get_int_parameter("int_multiple_space", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'int_multiple_space' is -2...");*/
  test_assert(ivalue == -2);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'double_tab' is available...");*/
  n = RUN_get_double_parameter("double_tab", &dvalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'double_tab' is -2.0...");*/
  test_assert(fabs(dvalue - -2.0) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  /* String parameters */

  n = 0;
  /* info("Checking 'string_parameter' is available...");*/
  n = RUN_get_string_parameter("string_parameter", string, 256);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'string_parameter' is 'ASCII'...");*/
  test_assert(strcmp(string, "ASCII") == 0);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'input_config' is available...");*/
  n = RUN_get_string_parameter("input_config", string, 256);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'input_config' is 'config.0'...");*/
  test_assert(strcmp(string, "config.0") == 0);
  /* info("yes\n");*/

  /* key_trail1 is 909; key_trail2 is 910 */

  n = RUN_get_int_parameter("key_trail1", &ivalue);
  assert(n == 1);
  assert(ivalue == 909);

  n = RUN_get_int_parameter("key_trail2", &ivalue);
  assert(n == 1);
  assert(ivalue == 910);

  /* Strings with trailing white space */

  n = RUN_get_string_parameter("key_trail3", string, 256);
  assert(n == 1);
  assert(strcmp(string, "string_3") == 0);

  n = RUN_get_string_parameter("key_trail4", string, 256);
  assert(n == 1);
  assert(strcmp(string, "string_4") == 0);

  /* Done. */

  n = 1;
  /* info("Checking all keys have been exhausted ...");*/
  n = RUN_get_active_keys();
  test_assert(n == 0);
  /* info("yes\n");*/

  pe_info(pe, "PASS     ./unit/test_runtime\n");
  pe_free(pe);

  return 0;
}

