/*****************************************************************************
 *
 *  io_info_args.h
 *
 *  This is a container to aggregate one lot of options for input, and
 *  one for output, and a small number of other quantities.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_INFO_ARGS_H
#define LUDWIG_IO_INFO_ARGS_H

#include "io_options_rt.h"

/* Specifies whether input or output is expected for given io type */

enum io_info_rw_enum {
  IO_INFO_NONE,
  IO_INFO_READ_ONLY,
  IO_INFO_WRITE_ONLY,
  IO_INFO_READ_WRITE
};

typedef enum   io_info_rw_enum io_info_rw_enum_t;
typedef struct io_info_args_s io_info_args_t;

struct io_info_args_s {
  io_options_t input;            /* Input mode, format, ... */
  io_options_t output;           /* Output mode, format, ... */
  int grid[3];                   /* Input and output have same grid */
  int iofreq;                    /* Output only. Frequency (every n steps) */
};

io_info_args_t io_info_args_default(void);
int io_info_args_iogrid_valid(int iogrid[3]);

#endif
