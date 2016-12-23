/*****************************************************************************
 *
 *  t_pe.c
 *
 *  Test the parallel environment.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "tests.h"

/*****************************************************************************
 *
 *  test_pe_suite
 *
 *****************************************************************************/

int test_pe_suite(void) {

  int my_rank = 0;
  int my_size = 1;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  assert(pe);

  test_assert(1);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);

  test_assert(pe_mpi_rank(pe) == my_rank);

  test_assert(pe_mpi_size(pe) == my_size);

  pe_info(pe, "PASS     ./unit/test_pe\n");
  pe_free(pe);

  return 0;
}
