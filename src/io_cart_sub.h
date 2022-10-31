/*****************************************************************************
 *
 *  io_cart_sub.h
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

#ifndef LUDWIG_IO_CART_SUB_H
#define LUDWIG_IO_CART_SUB_H

#include "coords.h"

typedef struct io_cart_sub_s io_cart_sub_t;

struct io_cart_sub_s {

  MPI_Comm parent;     /* Global Cartesian communictor */
  MPI_Comm comm;       /* MPI communicator for this 3d subsection */
  int size[3];         /* Global I/O Grid or topology */
  int coords[3];       /* Cartesian position of this rank in group */

  int ntotal[3];       /* Total sites (all files) */
  int nlocal[3];       /* Local sites (this file) */
  int offset[3];       /* Offset (this file) */

  int nfile;           /* Total number of files in decomposition */
  int index;           /* Index of this group {0, 1, ..., nfile-1} */
};

int io_cart_sub_create(cs_t * cs, int iogrid[3], io_cart_sub_t * iosub);
int io_cart_sub_free(io_cart_sub_t * iosub);
int io_cart_sub_printf(const io_cart_sub_t * iosub);

#endif
