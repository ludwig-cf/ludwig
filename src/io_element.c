/*****************************************************************************
 *
 *  io_element.c
 *
 *  Provides a container for i/o record description.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "io_element.h"

/*****************************************************************************
 *
 *  io_endian_from_string
 *
 *****************************************************************************/

io_endian_enum_t io_endian_from_string(const char * str) {

  io_endian_enum_t endianness = IO_ENDIAN_UNKNOWN;

  if (strcmp(str, "BIG_ENDIAN") == 0) endianness = IO_ENDIAN_BIG_ENDIAN;
  if (strcmp(str, "LITTLE_ENDIAN") == 0) endianness = IO_ENDIAN_LITTLE_ENDIAN;
  
  return endianness;
}

/*****************************************************************************
 *
 *  io_endian_to_string
 *
 *****************************************************************************/

const char * io_endian_to_string(io_endian_enum_t endianness) {

  const char * str = NULL;

  switch (endianness) {
  case IO_ENDIAN_LITTLE_ENDIAN:
    str = "LITTLE_ENDIAN";
    break;
  case IO_ENDIAN_BIG_ENDIAN:
    str = "BIG_ENDIAN";
    break;
  default:
    str = "UNKNOWN";
  }

  return str;
}

/*****************************************************************************
 *
 *  io_element_null
 *
 *  In particular, MPI_DATATYPE_NULL should be correct.
 *
 *****************************************************************************/

io_element_t io_element_null(void) {

  io_element_t element = {.datatype = MPI_DATATYPE_NULL,
                          .datasize = 0,
                          .count    = 0,
                          .endian   = IO_ENDIAN_UNKNOWN};

  return element;
}

/*****************************************************************************
 *
 *  io_element_to_json
 *
 *  Caller responisble for releasing resources.
 *
 *****************************************************************************/

int io_element_to_json(const io_element_t * element, cJSON ** json) {

  int ifail = 0;

  assert(element);

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {
    cJSON * myjson = cJSON_CreateObject();

    cJSON_AddStringToObject(myjson, "MPI_Datatype",
			    util_io_mpi_datatype_to_string(element->datatype));
    cJSON_AddNumberToObject(myjson, "Size (bytes)", element->datasize);
    cJSON_AddNumberToObject(myjson, "Count", element->count);
    cJSON_AddStringToObject(myjson, "Endianness",
			    io_endian_to_string(element->endian));
    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_element_from_json
 *
 *****************************************************************************/

int io_element_from_json(const cJSON * json, io_element_t * element) {

  int ifail = 0;

  if (json == NULL || element == NULL) {
    ifail = -1;
  }
  else {
    cJSON * datatype = cJSON_GetObjectItem(json, "MPI_Datatype");
    cJSON * datasize = cJSON_GetObjectItem(json, "Size (bytes)");
    cJSON * count = cJSON_GetObjectItem(json, "Count");
    cJSON * endianness = cJSON_GetObjectItem(json, "Endianness");

    if (datatype) {
      char * str = cJSON_GetStringValue(datatype);
      element->datatype = util_io_string_to_mpi_datatype(str);
    }

    if (datasize) element->datasize = cJSON_GetNumberValue(datasize);
    if (count)    element->count    = cJSON_GetNumberValue(count);

    if (endianness) {
      char * str = cJSON_GetStringValue(endianness);
      element->endian = io_endian_from_string(str);
    }
    /* Indicate what failed, if anything */
    if (!datatype)   ifail += 1;
    if (!datasize)   ifail += 2;
    if (!count)      ifail += 4;
    if (!endianness) ifail += 8;
  }

  return ifail;
}
