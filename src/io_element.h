/*****************************************************************************
 *
 *  io_element.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_ELEMENT_H
#define LUDWIG_IO_ELEMENT_H

#include "mpi.h"
#include "util_cjson.h"

/* Utility */

typedef enum {
  IO_ENDIAN_UNKNOWN,
  IO_ENDIAN_LITTLE_ENDIAN,
  IO_ENDIAN_BIG_ENDIAN
} io_endian_enum_t;

io_endian_enum_t io_endian_from_string(const char * str);
const char * io_endian_to_string(io_endian_enum_t endian);


typedef struct io_element_s io_element_t;

struct io_element_s {
  MPI_Datatype datatype;    /* Fundamental datatype */
  size_t datasize;          /* sizeof(datatype) */
  int count;                /* Items per record (scalar, vector, ...) */
  io_endian_enum_t endian;  /* Big, little. */
};

io_element_t io_element_null(void);
int io_element_from_json(const cJSON * json, io_element_t * element);
int io_element_to_json(const io_element_t * element, cJSON * json);

#endif
