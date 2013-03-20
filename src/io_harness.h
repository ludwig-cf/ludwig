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

struct io_info_t * io_info_create(void);
struct io_info_t * io_info_create_with_grid(const int *);
void io_info_destroy(struct io_info_t *);

void io_info_set_name(struct io_info_t *, const char *);
void io_info_set_write(struct io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_read(struct io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_bytesize(struct io_info_t * p, size_t);
void io_info_set_processor_independent(struct io_info_t *);
void io_info_set_processor_dependent(struct io_info_t *);

void io_info_set_read_ascii(struct io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_read_binary(struct io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_write_ascii(struct io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_write_binary(struct io_info_t *, int(*)(FILE *,int,int,int));
void io_info_set_format_ascii(struct io_info_t *);
void io_info_set_format_binary(struct io_info_t *);

void io_read(char *, struct io_info_t *);
void io_write(char *, struct io_info_t *);
void io_write_metadata(char *, struct io_info_t *);
void io_remove(char *, struct io_info_t *);
void io_info_single_file_set(struct io_info_t * info);
#endif
