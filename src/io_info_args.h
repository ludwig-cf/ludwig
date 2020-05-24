/*****************************************************************************
 *
 *  io_info_args.h
 *
 *  This is onlu a small container.
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

#ifndef LUDWIG_IO_INFO_ARGS_H
#define LUDWIG_IO_INFO_ARGS_H

#include "pe.h"
#include "io_options_rt.h"

typedef enum   io_info_rw_enum io_info_rw_enum_t;
typedef struct io_info_args_s io_info_args_t;

/* Specifies whether input or output is expected for given io type */

enum io_info_rw_enum {IO_INFO_NONE,
		      IO_INFO_READ_ONLY,
		      IO_INFO_WRITE_ONLY,
		      IO_INFO_READ_WRITE};

/* Container for run time arguments to allow io_info_t creation. */

struct io_info_args_s {
  io_options_t input;            /* Input mode, format, ... */
  io_options_t output;           /* Output mode, format, ... */
  int grid[3];                   /* Input and output have same grid */
  int nfreq;                     /* Output only. Frequency (every n steps) */
};

__host__ io_info_args_t io_info_args_default(void);
__host__ int io_info_args_iogrid_valid(int iogrid[3]);

#endif
