/*****************************************************************************
 *
 *  util_io.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_IO_H
#define LUDWIG_UTIL_IO_H

#include <mpi.h>

MPI_Datatype util_io_string_to_mpi_datatype(const char * str);
const char * util_io_mpi_datatype_to_string(MPI_Datatype dt);

#endif
