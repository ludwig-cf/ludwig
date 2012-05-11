/*****************************************************************************
 *
 *  io_harness.h
 *
 *  $Id: io_harness.h,v 1.2 2008-08-24 16:55:42 kevin Exp $
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

enum io_format_flag {IO_FORMAT_NULL,
                     IO_FORMAT_ASCII,
		     IO_FORMAT_BINARY,
                     IO_FORMAT_ASCII_SERIAL,
                     IO_FORMAT_BINARY_SERIAL,
                     IO_FORMAT_DEFAULT};

typedef struct io_info_s io_info_t;
typedef int (*io_rw_cb_ft)(io_info_t * obj, FILE * fp, int index, void * self);

io_info_t * io_info_create(void);
io_info_t * io_info_create_with_grid(const int *);
void io_info_destroy(io_info_t *);

void io_info_set_name(io_info_t *, const char *);
void io_info_set_write(io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_read(io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_bytesize(io_info_t * p, size_t);
void io_info_set_processor_independent(io_info_t *);
void io_info_set_processor_dependent(io_info_t *);

void io_info_set_read_ascii(io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_read_binary(io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_write_ascii(io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_write_binary(io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_format_ascii(io_info_t *);
void io_info_set_format_binary(io_info_t *);

void io_read(char *, io_info_t *);
void io_write(char *, io_info_t *);
void io_write_metadata(char *, io_info_t *);

int io_remove(char *, io_info_t *);
int io_info_format_set(io_info_t * obj, int form_in, int form_out); 
int io_info_format_in_set(io_info_t * obj, int form_in); 
int io_info_format_out_set(io_info_t * obj, int form_out); 

int io_info_read_set(io_info_t * obj, int format, io_rw_cb_ft);
int io_info_write_set(io_info_t * obj, int format, io_rw_cb_ft);

#endif
