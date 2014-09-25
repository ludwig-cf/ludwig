/*****************************************************************************
 *
 *  test_coords_field.h
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

#ifndef TEST_COORDS_FIELD_H
#define TEST_COORDS_FIELD_H

#include <mpi.h>

typedef int (* halo_ft)(int ic, int jc, int kc, int n, void * ref);

int test_coords_field_set(int nf, void * buf, MPI_Datatype mpidata,
			  halo_ft bufset);
int test_coords_field_check(int nhcomm, int nf, void * buf,
			    MPI_Datatype mpidata, halo_ft bufref);
int test_ref_char1(int ic, int jc, int kc, int n, void * buf);
int test_ref_double1(int ic, int jc, int kc, int n, void * buf);

#endif
