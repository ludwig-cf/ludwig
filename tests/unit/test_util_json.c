/*****************************************************************************
 *
 *  test_util_json.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "pe.h"
#include "util_bits.h"
#include "util_json.h"

int test_util_json_to_int_array(void);
int test_util_json_to_double_array(void);
int test_util_json_to_file(pe_t * pe);

/*****************************************************************************
 *
 *  test_util_json_suite
 *
 *****************************************************************************/

int test_util_json_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_json_to_int_array();
  test_util_json_to_double_array();
  test_util_json_to_file(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_json_to_int_array
 *
 *****************************************************************************/

int test_util_json_to_int_array(void) {

  int ifail = 0;

  {
    int array[3] = {1, 2, 3};
    cJSON * json = cJSON_CreateIntArray(array, 3);
    {
      int val[3] = {0};
      int iret = util_json_to_int_array(json, val, 3);
      if (iret != 3) ifail = -1;
      assert(val[0] == array[0]);
      assert(val[1] == array[1]);
      assert(val[2] == array[2]);
    }
    cJSON_Delete(json);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_json_to_double_array
 *
 *****************************************************************************/

int test_util_json_to_double_array(void) {

  int ifail = 0;

  {
    double array[3] = {1.0, 2.1, -3.2};
    cJSON * json = cJSON_CreateDoubleArray(array, 3);
    {
      double val[3] = {0};
      int iret = util_json_to_double_array(json, val, 3);
      if (iret != 3) ifail = -1;
      assert(util_double_same(val[0], array[0]));
      assert(util_double_same(val[1], array[1]));
      assert(util_double_same(val[2], array[2]));
    }
    cJSON_Delete(json);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_json_to_file
 *
 *  It's convenient to test util_json_from_file() at the same time.
 *
 *****************************************************************************/

int test_util_json_to_file(pe_t * pe) {

  int ifail = 0;
  char filename[BUFSIZ] = {0};

  /* Some specimin json */
  cJSON * jstr = cJSON_Parse("{\"Test\": \"data\"}");

  /* Write/read is entirely serial. So use per-process name ... */

  sprintf(filename, "test-util-json-to-file-%4.4d.json", pe_mpi_rank(pe));
  ifail = util_json_to_file(filename, jstr);
  assert(ifail == 0);

  /* Read (same file). We just test the return code and we have an object. */
  {
    cJSON * json = NULL;
    ifail = util_json_from_file(filename, &json);
    assert(ifail == 0);
    assert(json);
    cJSON_Delete(json);
  }

  remove(filename);
  cJSON_Delete(jstr);

  return ifail;
}
