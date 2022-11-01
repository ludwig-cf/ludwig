/*****************************************************************************
 *
 *  util_io.c
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

#include "util_io.h"

/*****************************************************************************
 *
 *  util_io_string_to_mpi_datatype
 *
 *  An unrecognised string will return MPI_TYPE_NULL.
 *
 *****************************************************************************/

MPI_Datatype util_io_string_to_mpi_datatype(const char * str) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;

  if (str == NULL) {
    dt = MPI_DATATYPE_NULL;
  }
  else if (strcmp(str, "MPI_CHAR") == 0) {
    dt = MPI_CHAR;
  }
  else if (strcmp(str, "MPI_SHORT") == 0) {
    dt = MPI_SHORT;
  }
  else if (strcmp(str, "MPI_INT") == 0) {
    dt = MPI_INT;
  }
  else if (strcmp(str, "MPI_LONG") == 0) {
    dt = MPI_LONG;
  }
  else if (strcmp(str, "MPI_UNSIGNED") == 0) {
    dt = MPI_UNSIGNED;
  }
  else if (strcmp(str, "MPI_UNSIGNED_CHAR") == 0) {
    dt = MPI_UNSIGNED_CHAR;
  }
  else if (strcmp(str, "MPI_UNSIGNED_SHORT") == 0) {
    dt = MPI_UNSIGNED_SHORT;
  }
  else if (strcmp(str, "MPI_UNSIGNED_LONG") == 0) {
    dt = MPI_UNSIGNED_LONG;
  }
  else if (strcmp(str, "MPI_FLOAT") == 0) {
    dt = MPI_DOUBLE;
  }
  else if (strcmp(str, "MPI_DOUBLE") == 0) {
    dt = MPI_DOUBLE;
  }
  else if (strcmp(str, "MPI_LONG_DOUBLE") == 0) {
    dt = MPI_LONG_DOUBLE;
  }
  else if (strcmp(str, "MPI_BYTE") == 0) {
    dt = MPI_BYTE;
  }
  else if (strcmp(str, "MPI_INT32_T") == 0) {
    dt = MPI_INT32_T;
  }
  else if (strcmp(str, "MPI_INT64_T") == 0) {
    dt = MPI_INT64_T;
  }
  else if (strcmp(str, "MPI_PACKED") == 0) {
    dt = MPI_PACKED;
  }

  return dt;
}

/*****************************************************************************
 *
 *  util_io_mpi_datatype_to_string
 *
 *  The string is exactly what one would expect for an intrinsic
 *  data type name. User defined types will (and unrecognised
 *  intrinsic types) will return MPI_TYPE_NULL.
 *
 *****************************************************************************/

const char * util_io_mpi_datatype_to_string(MPI_Datatype dt) {

  const char * str = NULL;

  if (dt == MPI_DATATYPE_NULL) {
    str = "MPI_DATATYPE_NULL";
  }
  else if (dt == MPI_CHAR) {
    str = "MPI_CHAR";
  }
  else if (dt == MPI_SHORT) {
    str = "MPI_SHORT";
  }
  else if (dt == MPI_INT) {
    str = "MPI_INT";
  }
  else if (dt == MPI_LONG) {
    str = "MPI_LONG";
  }
  else if (dt == MPI_UNSIGNED) {
    str = "MPI_UNSIGNED";
  }
  else if (dt == MPI_UNSIGNED_CHAR) {
    str = "MPI_UNSIGNED_CHAR";
  }
  else if (dt == MPI_UNSIGNED_SHORT) {
    str = "MPI_UNSIGNED_SHORT";
  }
  else if (dt == MPI_UNSIGNED_LONG) {
    str = "MPI_UNSIGNED_LONG";
  }
  else if (dt == MPI_FLOAT) {
    str = "MPI_FLOAT";
  }
  else if (dt == MPI_DOUBLE) {
    str = "MPI_DOUBLE";
  }
  else if (dt == MPI_LONG_DOUBLE) {
    str = "MPI_LONG_DOUBLE";
  }
  else if (dt == MPI_BYTE) {
    str = "MPI_BYTE";
  }
  else if (dt == MPI_INT32_T) {
    str = "MPI_INT32_T";
  }
  else if (dt == MPI_INT64_T) {
    str = "MPI_INT64_T";
  }
  else if (dt == MPI_PACKED) {
    str = "MPI_PACKED";
  }
  else {
    /* Unrecognised/Not included yet. */
    str = "MPI_DATATYPE_NULL";
  }

  assert(str != NULL);

  return str;
}
