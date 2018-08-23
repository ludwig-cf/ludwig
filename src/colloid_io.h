/*****************************************************************************
 *
 *  colloid_io.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_IO_H
#define LUDWIG_COLLOID_IO_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"

typedef struct colloid_io_s colloid_io_t;

int  colloid_io_create(pe_t * pe, cs_t * cs, int io_grid[3],
		       colloids_info_t * info,
		       colloid_io_t ** cio);
int colloid_io_free(colloid_io_t * cio);
int colloid_io_info_set(colloid_io_t * cio, colloids_info_t * info);
int colloid_io_info(colloid_io_t * cio);

int colloid_io_read(colloid_io_t * cio, const char * filename);
int colloid_io_write(colloid_io_t * cio, const char * filename);
int colloid_io_format_input_ascii_set(colloid_io_t * cio);
int colloid_io_format_input_binary_set(colloid_io_t * cio);
int colloid_io_format_input_serial_set(colloid_io_t * cio);
int colloid_io_format_output_ascii_set(colloid_io_t * cio);
int colloid_io_format_output_binary_set(colloid_io_t * cio);

#endif
