/*****************************************************************************
 *
 *  util_io.h
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_IO_H
#define LUDWIG_UTIL_IO_H

#include <mpi.h>

MPI_Datatype util_io_string_to_mpi_datatype(const char * str);
const char * util_io_mpi_datatype_to_string(MPI_Datatype dt);

#endif
