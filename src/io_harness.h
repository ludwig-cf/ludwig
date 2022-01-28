/*****************************************************************************
 *
 *  io_harness.h
 *
 *  I/O control for general lattice based fields.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2007-2020 The University of Edinburgh
 *
 *  Contributin authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_HARNESS_H
#define LUDWIG_IO_HARNESS_H

#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "io_options.h"
#include "io_info_args.h"

typedef enum io_format_enum {IO_FORMAT_NULL,
			     IO_FORMAT_ASCII,
			     IO_FORMAT_BINARY,
			     IO_FORMAT_ASCII_SERIAL,
			     IO_FORMAT_BINARY_SERIAL,
			     IO_FORMAT_DEFAULT} io_format_enum_t;

typedef struct io_implementation_s io_implementation_t;
typedef struct io_info_s io_info_t;

/* Callback signature for lattice site I/O */
typedef int (*io_rw_cb_ft)(FILE * fp, int index, void * self);

struct io_implementation_s {
  char         name[BUFSIZ];      /* Descriptive name */
  io_rw_cb_ft  write_ascii;       /* Callback function for ascii write */
  io_rw_cb_ft  write_binary;      /* Callback function for binary write */
  io_rw_cb_ft  read_ascii;        /* Callback function for ascii read */
  io_rw_cb_ft  read_binary;       /* Callback function for binary read */
  size_t       bytesize_ascii;    /* Bytes per ascii read */
  size_t       bytesize_binary;   /* Bytes per binary read */
};

typedef struct io_decomposition_s io_decomposition_t;

struct io_decomposition_s {
  int n_io;         /* Total number of I/O groups (files) in decomposition */
  int index;        /* Index of this I/O group {0, 1, ...} */
  MPI_Comm xcomm;   /* Cross-communicator for same rank in different groups */
  MPI_Comm comm;    /* MPI communicator for this group */
  int rank;         /* Rank of this process in group communicator */
  int size;         /* Size of this group in processes */
  int ngroup[3];    /* Global I/O group topology XYZ */
  int coords[3];    /* Coordinates of this group in I/O topology XYZ */
  int nsite[3];     /* Size of file in lattice sites */
  int offset[3];    /* Offset of the file on the lattice */
};

struct io_info_s {
  pe_t * pe;
  cs_t * cs;

  io_info_args_t args;
  io_implementation_t impl;
  io_decomposition_t * comm;

  io_decomposition_t * io_comm;
  size_t bytesize;                   /* Actual output per site */
  size_t bytesize_ascii;             /* ASCII line size */
  size_t bytesize_binary;            /* Binary record size */
  int nsites;                        /* No. sites this group */
  int maxlocal;                      /* Max. no. sites per rank this group */
  int metadata_written;
  int processor_independent;
  int single_file_read;
  int report;                        /* Report time taken for output */
  char metadata_stub[FILENAME_MAX];
  char name[FILENAME_MAX];
  io_rw_cb_ft write_data;
  io_rw_cb_ft write_ascii;
  io_rw_cb_ft write_binary;
  io_rw_cb_ft read_data;
  io_rw_cb_ft read_ascii;
  io_rw_cb_ft read_binary;
};

__host__ int io_info_create(pe_t * pe, cs_t * cs, io_info_args_t * arg,
			    io_info_t ** pinfo);
__host__ int io_info_free(io_info_t *);

__host__ int io_info_create_impl(pe_t * pe, cs_t * cs, io_info_args_t arg,
				 const io_implementation_t * impl,
				 io_info_t ** info);

__host__ int io_info_input_bytesize(io_info_t * info, size_t * bs);
__host__ int io_info_output_bytesize(io_info_t * info, size_t * bs);

__host__ void io_info_set_name(io_info_t *, const char *);
__host__ void io_info_set_write(io_info_t *, int (*) (FILE *, int, int, int));
__host__ void io_info_set_read(io_info_t *, int (*) (FILE *, int, int, int));
__host__ void io_info_set_processor_independent(io_info_t *);
__host__ void io_info_set_processor_dependent(io_info_t *);
__host__ void io_info_single_file_set(io_info_t * info);

__host__ int io_info_set_bytesize(io_info_t * p, io_format_enum_t t, size_t);
__host__ int io_write_metadata(io_info_t * info);
__host__ int io_write_metadata_file(io_info_t * info, char * filestub);
__host__ int io_info_metadata_filestub_set(io_info_t * info, const char * filestub);

__host__ int io_remove(const char * filename_stub, io_info_t * obj);
__host__ int io_remove_metadata(io_info_t * obj, const char * file_stub);
__host__ int io_info_format_set(io_info_t * obj, int form_in, int form_out); 
__host__ int io_info_format_in_set(io_info_t * obj, int form_in); 
__host__ int io_info_format_out_set(io_info_t * obj, int form_out); 

__host__ int io_info_read_set(io_info_t * obj, int format, io_rw_cb_ft);
__host__ int io_info_write_set(io_info_t * obj, int format, io_rw_cb_ft);
__host__ int io_write_data(io_info_t * obj, const char * filename_stub, void * data);
__host__ int io_read_data(io_info_t * obj, const char * filename_stub, void * data);

#endif
