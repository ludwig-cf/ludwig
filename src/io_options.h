/*****************************************************************************
 *
 *  io_options.h
 *
 *  Container for lattice-related I/O options.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_OPTIONS_H
#define LUDWIG_IO_OPTIONS_H

#include "pe.h"

/*
 *  I/O Modes:
 *
 *  IO_MODE_SINGLE:    single file with decomposition independent ordering;
 *                     data written as if in serial.
 *  IO_MODE_MULTIPLE:  one or more files with decomposition dependent order;
 *                     output must be post-processed to recover serial order.
 */

enum io_mode_enum {IO_MODE_INVALID, IO_MODE_SINGLE, IO_MODE_MULTIPLE};

/*
 * Record formats:
 *
 * IO_RECORD_ASCII
 * IO_RECORD_BINARY
 *
 */

enum io_rformat_enum {IO_RECORD_INVALID, IO_RECORD_ASCII, IO_RECORD_BINARY};

/*
 * Metadata versions:
 *
 * IO_METADATA_SINGLE_V1
 * IO_METADATA_MULTI_V1
 *
 */

enum io_metadata_enum {IO_METADATA_INVALID,
		       IO_METADATA_SINGLE_V1, IO_METADATA_MULTI_V1};

/* Here is the options container type */

typedef enum io_mode_enum     io_mode_enum_t;
typedef enum io_rformat_enum  io_rformat_enum_t;
typedef enum io_metadata_enum io_metadata_enum_t;

struct io_options_s {
  io_mode_enum_t     mode;              /* Single file, multiple file */
  io_rformat_enum_t  iorformat;         /* Record format ascii/binary */
  io_metadata_enum_t metadata_version;  /* Metadata version no. */
  int                report;            /* Switch reporting on or off */
  int                asynchronous;      /* Not implemented at the moment */
};

typedef struct io_options_s io_options_t;


/* A convenience to define/initialise default values. */

#define IO_MODE_DEFAULT()             IO_MODE_SINGLE
#define IO_RECORD_FORMAT_DEFAULT()    IO_RECORD_BINARY
#define IO_METADATA_VERSION_DEFAULT() IO_METADATA_SINGLE_V1
#define IO_OPTIONS_DEFAULT()         {IO_MODE_DEFAULT(), \
                                      IO_RECORD_FORMAT_DEFAULT(), \
                                      IO_METADATA_VERSION_DEFAULT(), 0, 0}

__host__ int io_options_valid(const io_options_t * options);
__host__ int io_options_mode_valid(io_mode_enum_t mode);
__host__ int io_options_rformat_valid(io_rformat_enum_t iorformat);
__host__ int io_options_metadata_version_valid(const io_options_t * options);

#endif
