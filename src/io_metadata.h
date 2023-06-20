/*****************************************************************************
 *
 *  io_metadata.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_METADATA_H
#define LUDWIG_IO_METADATA_H

#include "pe.h"
#include "coords.h"
#include "cs_limits.h"
#include "io_options.h"
#include "io_element.h"
#include "io_subfile.h"
#include "util_json.h"

typedef struct io_metadata_s io_metadata_t;

struct io_metadata_s {

  cs_t * cs;                         /* Keep a reference to coordinates */
  cs_limits_t limits;                /* Always local size with no halo */
  MPI_Comm parent;                   /* Cartesian communicator */
  MPI_Comm comm;                     /* Cartesian sub-communicator */
  int iswriten;                      /* updated to true if file is writen */

  io_options_t options;
  io_element_t element;
  io_subfile_t subfile;

};

int io_metadata_create(cs_t * cs,
                       const io_options_t * options,
                       const io_element_t * element,
                       io_metadata_t ** metadata);
int io_metadata_free(io_metadata_t ** metadata);
int io_metadata_initialise(cs_t * cs,
			   const io_options_t * options,
			   const io_element_t * element,
			   io_metadata_t * metadata);
int io_metadata_finalise(io_metadata_t * metadata);

int io_metadata_to_json(const io_metadata_t * metadata, cJSON ** json);
int io_metadata_from_json(cs_t * cs, const cJSON * json,
			  io_metadata_t * metadata);

int io_metadata_write(const io_metadata_t * metadata,
		      const char * stub,
		      const char * extra,
		      const cJSON * json);
int io_metadata_from_file(pe_t * pe, const char * filename,
			  io_metadata_t ** metadata);

#endif
