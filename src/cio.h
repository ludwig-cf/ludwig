/*****************************************************************************
 *
 *  cio.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef CIO_H
#define CIO_H

void colloid_io_init(int io_grid[3]);
void colliod_io_finish(void);

void colloid_io_read(const char * filename);
void colloid_io_write(const char * filename);

void colloid_io_format_input_ascii_set(void);
void colloid_io_format_input_binary_set(void);
void colloid_io_format_input_serial_set(void);
void colloid_io_format_output_ascii_set(void);
void colloid_io_format_output_binary_set(void);

#endif
