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

int io_element_to_json(const io_element_t * element, cJSON * json) {

  int ifail = 0;

  assert(element);

  if (json != NULL) {
    ifail = -1;
  }
  else {
    json = cJSON_CreateObject();

    cJSON_AddStringToObject(json, "MPI_Datatype", "MPI_DOUBLE TBC");
    cJSON_AddNumberToObject(json, "Datasize (bytes)", element->datasize);
    cJSON_AddNumberToObject(json, "Count", element->count);
    cJSON_AddStringToObject(json, "Endianness",
			    io_endian_to_string(element->endian));
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

  assert(element);

  if (json == NULL) {
    ifail = -1;
  }
  else {
    cJSON * datatype = cJSON_GetObjectItem(json, "MPI_Datatype");
    cJSON * datasize = cJSON_GetObjectItem(json, "Datasize (bytes)");
    cJSON * count = cJSON_GetObjectItem(json, "Count");
    cJSON * endianness = cJSON_GetObjectItem(json, "Endianness");

    if (datatype) {
      /* Another string to be converted to MPI_Datatype */
      element->datatype = MPI_DOUBLE; /* TBC */
    }
    if (datasize) element->datasize = cJSON_GetNumberValue(datasize);
    if (count)    element->count    = cJSON_GetNumberValue(count);

    if (endianness) {
      char * str = {0};
      str = cJSON_GetStringValue(endianness);
      element->endian = io_endian_from_string(str);
    }
  }

  return ifail;
}
