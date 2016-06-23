/*****************************************************************************
 *
 *  field_s.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FIELD_S_H
#define FIELD_S_H

#include <mpi.h>

#include "io_harness.h"
#include "memory.h"
#include "halo_swap.h"
#include "field.h"

struct field_s {
  int nf;                       /* Number of field components */
  int nhcomm;                   /* Halo width required */
  int nsites;                   /* Local sites (allocated) */
  double * data;                /* Field data */

  io_info_t * info;             /* I/O Handler */
  char * name;                  /* "phi", "p", "q" etc. */
  halo_swap_t * halo;           /* Halo swap driver object */

  field_t * target;             /* target structure */ 
};

#define addr_qab(nsites, index, ia) addr_rank1(nsites, NQAB, index, ia)
#define vaddr_qab(nsites, index, ia, iv) vaddr_rank1(nsites, NQAB, index, ia, iv)

#endif

