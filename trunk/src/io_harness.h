/*****************************************************************************
 *
 *  io_harness.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef IOHARNESS_H
#define IOHARNESS_H

#include <stdio.h>
#include "targetDP.h"

enum io_format_flag {IO_FORMAT_NULL,
                     IO_FORMAT_ASCII,
		     IO_FORMAT_BINARY,
                     IO_FORMAT_ASCII_SERIAL,
                     IO_FORMAT_BINARY_SERIAL,
                     IO_FORMAT_DEFAULT};

typedef struct io_info_s io_info_t;
typedef int (*io_rw_cb_ft)(FILE * fp, int index, void * self);

HOST io_info_t * io_info_create(void);
HOST io_info_t * io_info_create_with_grid(const int *);
HOST void io_info_destroy(io_info_t *);

HOST void io_info_set_name(io_info_t *, const char *);
HOST void io_info_set_write(io_info_t *, int (*) (FILE *, int, int, int));
HOST void io_info_set_read(io_info_t *, int (*) (FILE *, int, int, int));
HOST void io_info_set_bytesize(io_info_t * p, size_t);
HOST void io_info_set_processor_independent(io_info_t *);
HOST void io_info_set_processor_dependent(io_info_t *);
HOST void io_info_single_file_set(io_info_t * info);

HOST int io_write_metadata(io_info_t * info);
HOST int io_write_metadata_file(io_info_t * info, char * filestub);
HOST int io_info_metadata_filestub_set(io_info_t * info, char * filestub);

HOST int io_remove(char * filename_stub, io_info_t * obj);
HOST int io_remove_metadata(io_info_t * obj, const char * file_stub);
HOST int io_info_format_set(io_info_t * obj, int form_in, int form_out); 
HOST int io_info_format_in_set(io_info_t * obj, int form_in); 
HOST int io_info_format_out_set(io_info_t * obj, int form_out); 

HOST int io_info_read_set(io_info_t * obj, int format, io_rw_cb_ft);
HOST int io_info_write_set(io_info_t * obj, int format, io_rw_cb_ft);
HOST int io_write_data(io_info_t * obj, const char * filename_stub, void * data);
HOST int io_read_data(io_info_t * obj, const char * filename_stub, void * data);

#endif
