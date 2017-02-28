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
 *  (c) 2012-2017 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COORDS_FIELD_H
#define LUDWIG_COORDS_FIELD_H

#include "coords.h"

__host__ int coords_field_init_mpi_indexed(cs_t * cs, int nhcomm, int nf,
					   MPI_Datatype mpidata,
					   MPI_Datatype halo[3]);

__host__ int coords_field_halo_rank1(cs_t * cs, int nall, int nhcomm, int na,
				     void * buf,
				     MPI_Datatype mpidata);
#endif
