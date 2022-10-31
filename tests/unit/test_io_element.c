/*****************************************************************************
 *
 *  test_io_element.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "io_element.h"

int test_io_endian_from_string(void);
int test_io_endian_to_string(void);
int test_io_element_null(void);
int test_io_element_from_json(void);
int test_io_element_to_json(void);

/*****************************************************************************
 *
 *  test_io_element_suite
 *
 *****************************************************************************/

int test_io_element_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_io_endian_from_string();
  test_io_endian_to_string();

  test_io_element_null();
  test_io_element_from_json();
  test_io_element_to_json();

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_endian_from_string
 *
 *****************************************************************************/

int test_io_endian_from_string(void) {

  {
    const char * str = "LITTLE_ENDIAN";
    io_endian_enum_t endian = io_endian_from_string(str);
    assert(endian == IO_ENDIAN_LITTLE_ENDIAN);
  }

  {
    const char * str = "BIG_ENDIAN";
    io_endian_enum_t endian = io_endian_from_string(str);
    assert(endian == IO_ENDIAN_BIG_ENDIAN);
  }

  {
    const char * str = "RUBBISH";
    io_endian_enum_t endian = io_endian_from_string(str);
    assert(endian == IO_ENDIAN_UNKNOWN);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_io_endian_to_string
 *
 *****************************************************************************/

int test_io_endian_to_string(void) {

  {
    io_endian_enum_t endian = IO_ENDIAN_UNKNOWN;
    const char * str = io_endian_to_string(endian);
    assert(strcmp(str, "UNKNOWN") == 0);
  }

  {
    io_endian_enum_t endian = IO_ENDIAN_LITTLE_ENDIAN;
    const char * str = io_endian_to_string(endian);
    assert(strcmp(str, "LITTLE_ENDIAN") == 0);
  }

  {
    io_endian_enum_t endian = IO_ENDIAN_BIG_ENDIAN;
    const char * str = io_endian_to_string(endian);
    assert(strcmp(str, "BIG_ENDIAN") == 0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_io_element_null
 *
 *****************************************************************************/

int test_io_element_null(void) {

  io_element_t element = io_element_null();

  assert(element.datatype == MPI_DATATYPE_NULL);
  assert(element.datasize == 0);
  assert(element.count    == 0);
  assert(element.endian   == IO_ENDIAN_UNKNOWN);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_element_from_json
 *
 *****************************************************************************/

int test_io_element_from_json(void) {

  int ifail = 0;

  {
    /* Null JSON object is a fail */
    cJSON * json = NULL;
    io_element_t element = io_element_null();
    ifail = io_element_from_json(json, &element);
    assert(ifail == -1);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_element_to_json
 *
 *****************************************************************************/

int test_io_element_to_json(void) {

  int ifail = 0;

  {
    io_element_t element = {.datatype = MPI_DOUBLE,
                            .datasize = 1,
                            .count    = 3,
                            .endian   = IO_ENDIAN_LITTLE_ENDIAN};
    cJSON * json = NULL;
    ifail = io_element_to_json(&element, json);

    cJSON_Delete(json);
  }

  return ifail;
}
