/*****************************************************************************
 *
 *  colloid_io.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2013 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOID_IO_H
#define COLLOID_IO_H

#include "colloids.h"

typedef struct colloid_io_s colloid_io_t;

int  colloid_io_create(int io_grid[3], colloids_info_t * info,
		       colloid_io_t ** cio);
void colliod_io_free(colloid_io_t * cio);

int colloid_io_read(colloid_io_t * cio, const char * filename);
int colloid_io_write(colloid_io_t * cio, const char * filename);
int colloid_io_format_input_ascii_set(colloid_io_t * cio);
int colloid_io_format_input_binary_set(colloid_io_t * cio);
int colloid_io_format_input_serial_set(colloid_io_t * cio);
int colloid_io_format_output_ascii_set(colloid_io_t * cio);
int colloid_io_format_output_binary_set(colloid_io_t * cio);

#endif
