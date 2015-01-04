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
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FIELD_S_H
#define FIELD_S_H

#include <mpi.h>
#include "field.h"

struct field_s {
  coords_t * cs;                /* Reference to coordinate system */
  le_t * le;                    /* Reference to Lees Edwards coordinates */
  int nf;                       /* Number of field components */
  int nhcomm;                   /* Halo width required */
  int nsites;                   /* Total sites (dependent on LE) */
  double * data;                /* Data on host */
  double * t_data;              /* Data on target */
  char * siteMask;              /* boolean lattice-shaped struture for masking (on host)*/
  char * t_siteMask;            /* boolean lattice-shaped struture for masking (on target)*/

  MPI_Datatype halo[3];         /* Halo exchange data types */
  io_info_t * info;             /* I/O Handler */
  char * name;                  /* "phi", "p", "q" etc. */
};

#endif
