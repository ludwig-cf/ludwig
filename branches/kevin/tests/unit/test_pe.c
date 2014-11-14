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
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

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

  pe_init();

  /* info("Checking pe_init() (post-hoc)...");*/
  test_assert(1);
  /*info("ok\n");*/

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);

  /* info("Checking pe_rank() is correct...");*/
  test_assert(pe_rank() == my_rank);
  /* info("yes\n");*/

  /* info("Checking pe_size() is correct...");*/
  test_assert(pe_size() == my_size);
  /* info("yes\n");*/

  /* verbose("Checking verbose()\n");*/

  /*info("Verbose working ok...");
  test_assert(1);
  info("yes\n");*/

  info("PASS     ./unit/test_pe\n");
  /* info("About to do pe_finalise()...\n");*/
  pe_finalise();

  return 0;
}
