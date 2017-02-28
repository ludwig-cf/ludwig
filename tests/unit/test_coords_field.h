/*****************************************************************************
 *
 *  test_coords_field.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef TEST_COORDS_FIELD_H
#define TEST_COORDS_FIELD_H

#include <mpi.h>
#include "coords.h"

typedef int (* halo_ft)(cs_t * cs, int ic, int jc, int kc, int n, void * ref);

int test_coords_field_set(cs_t * cs, int nf, void * buf, MPI_Datatype mpidata,
			  halo_ft bufset);
int test_coords_field_check(cs_t * cs, int nhcomm, int nf, void * buf,
			    MPI_Datatype mpidata, halo_ft bufref);

int test_ref_char1(cs_t * cs, int ic, int jc, int kc, int n, void * buf);
int test_ref_double1(cs_t * cs, int ic, int jc, int kc, int n, void * buf);

#endif
