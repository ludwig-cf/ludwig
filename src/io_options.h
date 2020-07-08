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

/* Record formats: */

enum io_record_format_enum {IO_RECORD_INVALID,
			    IO_RECORD_ASCII,
			    IO_RECORD_BINARY};

/* Metadata versions: */

enum io_metadata_version_enum {IO_METADATA_INVALID,
			       IO_METADATA_SINGLE_V1,
			       IO_METADATA_MULTI_V1};

/* Options container type */

typedef enum io_mode_enum             io_mode_enum_t;
typedef enum io_record_format_enum    io_record_format_enum_t;
typedef enum io_metadata_version_enum io_metadata_version_enum_t;

struct io_options_s {
  io_mode_enum_t             mode;             /* Single file, multiple file */
  io_record_format_enum_t    iorformat;        /* Record format ascii/binary */
  io_metadata_version_enum_t metadata_version; /* Metadata version no. */
  int                        report;           /* Switch reporting on/off */
  int                        asynchronous;     /* Not implemented */
};

typedef struct io_options_s io_options_t;

__host__ io_mode_enum_t io_mode_default(void);
__host__ io_record_format_enum_t io_record_format_default(void);
__host__ io_metadata_version_enum_t io_metadata_version_default(void);
__host__ io_options_t io_options_default(void);

__host__ int io_options_valid(const io_options_t * options);
__host__ int io_options_mode_valid(io_mode_enum_t mode);
__host__ int io_options_record_format_valid(io_record_format_enum_t iorformat);
__host__ int io_options_metadata_version_valid(const io_options_t * options);

#endif
