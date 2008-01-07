/*****************************************************************************
 *
 *  io_harness.h
 *
 *  $Id: io_harness.h,v 1.1.2.1 2008-01-07 17:32:29 kevin Exp $
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

void io_init(void);
void io_finalise(void);

struct io_info_t * io_info_create(void);
void io_info_destroy(struct io_info_t *);
struct io_decomposition_t * io_decomposition_create(void);
void io_decomposition_destroy(struct io_decomposition_t *);

void io_info_set_name(struct io_info_t *, const char *);
void io_info_set_write(struct io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_read(struct io_info_t *, int (*) (FILE *, int, int, int));
void io_info_set_bytesize(struct io_info_t * p, size_t);
void io_info_set_decomposition(struct io_info_t *, const int);

void io_read(char *, struct io_info_t *);
void io_write(char *, struct io_info_t *);

#endif
