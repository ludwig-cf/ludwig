/*****************************************************************************
 *
 *  test_kernel.c
 *
 *  For the wrapper kernel.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "kernel.h"


/*****************************************************************************
 *
 *  test_kernel_suite
 *
 *****************************************************************************/

__host__ int test_kernel_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  pe_info(pe, "PASS     ./unit/test_kernel\n");

  pe_free(pe);

  return 0;
}
