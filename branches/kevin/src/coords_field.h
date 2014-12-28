/*****************************************************************************
 *
 *  coords_field.h
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

#ifndef COORDS_FIELD_H
#define COORDS_FIELD_H

#include <mpi.h>
#include "coords.h"

__host__ int coords_field_index(int index, int n, int nf, int * indexf);
__host__ int coords_field_init_mpi_indexed(coords_t * cs, int nhcomm, int nf,
					   MPI_Datatype mpidata,
					   MPI_Datatype halo[3]);
__host__ int coords_field_halo(coords_t * cs, int nhcomm, int nf, void * buf,
			       MPI_Datatype mpidata,
			       MPI_Datatype halo[3]);

#endif
