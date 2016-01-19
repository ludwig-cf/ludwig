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
#include "field.h"

struct field_s {
  int nf;                       /* Number of field components */
  int nhcomm;                   /* Halo width required */
  double * data;                /* Data on host */
  double * t_data;              /* Data on target: DEPRECATED (see tcopy below)*/

  MPI_Datatype halo[3];         /* Halo exchange data types */
  io_info_t * info;             /* I/O Handler */
  char * name;                  /* "phi", "p", "q" etc. */

  field_t * tcopy;              /* copy of this structure on target */ 

};

#ifndef OLD_SHIT

#include "memory.h"

#define addr_qab(nsites, index, ia) addr_rank1(nsites, NQAB, index, ia)
#define vaddr_qab(nsites, index, ia, iv) vaddr_rank1(nsites, NQAB, index, ia, iv)

#else

#ifdef LB_DATA_SOA
#define FLDADR ADDR_VECSITE_R
#else
#define FLDADR ADDR_VECSITE
#endif

#endif

#endif
