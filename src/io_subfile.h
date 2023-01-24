/*****************************************************************************
 *
 *  io_subfile.h
 *
 *  Description of i/o decomposition of the whole system into Cartesian
 *  blocks with one block per file.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_SUBFILE_H
#define LUDWIG_IO_SUBFILE_H

#include "coords.h"
#include "util_json.h"

typedef struct io_subfile_s io_subfile_t;

struct io_subfile_s {

  /* Data shared between all ranks in the current block/file */

  int nfile;           /* Number of blocks/files */
  int index;           /* Index {0 .. nfile-1} */
  int iosize[3];       /* 3-d (Cartesian) decomposition extents */
  int coords[3];       /* Carteisan position in decomposition */

  int ndims;           /* System dimensions (always 3 here) */
  int sizes[3];        /* Total size of block/file in lattice sites */
  int offset[3];       /* Offset of block/file in system (sites) */
};

io_subfile_t io_subfile_default(cs_t * cs);
int io_subfile_create(cs_t * cs, const int iogrid[3], io_subfile_t * subfile);
int io_subfile_to_json(const io_subfile_t * subfile, cJSON ** json);
int io_subfile_from_json(const cJSON * json, io_subfile_t * subfile);
int io_subfile_name(const io_subfile_t * subfile, const char * stub, int it,
		    char * filename, size_t bufsz);

#endif
