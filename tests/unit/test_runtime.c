/*****************************************************************************
 *
 *  Test the runtime interface.
 *
 *  Associated test input files are:
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "tests.h"

int test_rt_general(pe_t * pe);
int test_rt_nvector(pe_t * pe);

/*****************************************************************************
 *
 *  test_rt_suite
 *
 *****************************************************************************/

int test_rt_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_rt_general(pe);
  test_rt_nvector(pe);

  pe_info(pe, "PASS     ./unit/test_runtime\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_rt_general
 *
 *****************************************************************************/

int test_rt_general(pe_t * pe) {

  int    n;
  int    ivalue;
  int    ivector[3];
  double dvalue;
  double dvector[3];
  char   string[256];

  rt_t * rt = NULL;

  assert(pe);

  rt_create(pe, &rt);
  rt_read_input_file(rt, "test_runtime_input1");

  /* info("Testing runtime.c...\n");

     info("Checking can read the file 'test_runtime_input1'... \n"); */

  test_assert(1);
  /* info("...input file read ok.\n");*/

  n = 0;
  /* info("Checking number of keys available is now 11... ");*/
  rt_active_keys(rt, &n);
  test_assert(n == 15);
  /* info("yes\n");*/

  n = rt_key_required(rt, "int_scalar", RT_FATAL);
  assert(n == 0);
  n = rt_key_required(rt, "int_not_present", RT_NONE);
  assert(n != 0);

  n = 0;
  /* info("Checking key 'int_scalar' is available...");*/
  n = rt_int_parameter(rt, "int_scalar", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking key 'int_scalar' has correct value...");*/
  test_assert(ivalue == 999);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'double_scalar' is available...");*/
  n = rt_double_parameter(rt, "double_scalar", &dvalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking key 'double_scalar' has correct value...");*/
  test_assert(fabs(dvalue - 3.33) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'temperature' is available...");*/
  n = rt_int_parameter(rt, "temperature", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'temperature' is -1...");*/
  test_assert(ivalue == -1);
  /*info("yes\n");*/

  n = 0;
  /* info("Checking 'temp' is available...");*/
  n = rt_int_parameter(rt, "temp", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'temp' is +1...");*/
  test_assert(ivalue == 1);
  /* info("yes\n");*/

  /* info("Checking 'temper' is 0...");*/
  n = rt_int_parameter(rt, "temper", &ivalue);
  test_assert(ivalue == 0);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking key 'int_vector' is available...");*/
  n = rt_int_parameter_vector(rt, "int_vector", ivector);
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
  n = rt_double_parameter_vector(rt, "double_vector", dvector);
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
  n = rt_int_parameter(rt, "int_dummy", &ivalue);
  test_assert(n == 0);
  /* info("ok\n");*/

  /* info("Checking 'double_dummy' does not exist ...");*/
  n = rt_double_parameter(rt, "double_dummy", &dvalue);
  test_assert(n == 0);
  /* info("ok\n");*/

  /* Parameters specified in odd syntax */

  n = 0;
  /* info("Checking 'int_multiple_space' is available...");*/
  n = rt_int_parameter(rt, "int_multiple_space", &ivalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'int_multiple_space' is -2...");*/
  test_assert(ivalue == -2);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'double_tab' is available...");*/
  n = rt_double_parameter(rt, "double_tab", &dvalue);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'double_tab' is -2.0...");*/
  test_assert(fabs(dvalue - -2.0) < TEST_DOUBLE_TOLERANCE);
  /* info("yes\n");*/

  /* String parameters */

  n = 0;
  /* info("Checking 'string_parameter' is available...");*/
  n = rt_string_parameter(rt, "string_parameter", string, 256);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'string_parameter' is 'ASCII'...");*/
  test_assert(strcmp(string, "ASCII") == 0);
  /* info("yes\n");*/

  n = 0;
  /* info("Checking 'input_config' is available...");*/
  n = rt_string_parameter(rt, "input_config", string, 256);
  test_assert(n == 1);
  /* info("yes\n");*/

  /* info("Checking 'input_config' is 'config.0'...");*/
  test_assert(strcmp(string, "config.0") == 0);
  /* info("yes\n");*/

  /* key_trail1 is 909; key_trail2 is 910 */

  n = rt_int_parameter(rt, "key_trail1", &ivalue);
  assert(n == 1);
  assert(ivalue == 909);

  n = rt_int_parameter(rt, "key_trail2", &ivalue);
  assert(n == 1);
  assert(ivalue == 910);

  /* Strings with trailing white space */

  n = rt_string_parameter(rt, "key_trail3", string, 256);
  assert(n == 1);
  assert(strcmp(string, "string_3") == 0);

  n = rt_string_parameter(rt, "key_trail4", string, 256);
  assert(n == 1);
  assert(strcmp(string, "string_4") == 0);

  /* Done. */

  n = 1;
  /* info("Checking all keys have been exhausted ...");*/
  rt_active_keys(rt, &n);
  test_assert(n == 0);
  /* info("yes\n");*/

  rt_free(rt);

  return 0;
}

/*****************************************************************************
 *
 *  test_rt_nvector
 *
 *****************************************************************************/

int test_rt_nvector(pe_t * pe) {

  int key_ret = 0;
  rt_t * rt = NULL;

  rt_create(pe, &rt);
  rt_add_key_value(rt, "ki2", "1_2");
  rt_add_key_value(rt, "kd4", "1.0_2.0_3.0_4.0");
  rt_add_key_value(rt, "bad_val", "1_x");

  {
    int i2[2] = {};
    key_ret = rt_int_nvector(rt, "ki2", 2, i2, RT_NONE);
    assert(key_ret == 0);
    assert(i2[0] == 1);
    assert(i2[1] == 2);
  }

  {
    int i3[3] = {};
    key_ret = rt_int_nvector(rt, "ki2", 3, i3, RT_NONE); /* Wrong length */
    assert(key_ret != 0);
  }

  {
    double v4[4] = {};
    key_ret = rt_double_nvector(rt, "kd4", 4, v4, RT_NONE);
    assert(key_ret == 0);
    assert(fabs(v4[0] - 1.0) < DBL_EPSILON);
    assert(fabs(v4[1] - 2.0) < DBL_EPSILON);
    assert(fabs(v4[2] - 3.0) < DBL_EPSILON);
    assert(fabs(v4[3] - 4.0) < DBL_EPSILON);
  }

  {
    int i2[2] = {};

    key_ret = rt_int_nvector(rt, "bad_val", 2, i2, RT_NONE); /* bad value */
    assert(key_ret != 0);
  }

  rt_free(rt);

  return key_ret;
}
