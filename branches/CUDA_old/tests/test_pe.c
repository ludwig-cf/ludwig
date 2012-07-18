/*****************************************************************************
 *
 *  t_pe.c
 *
 *  Test the parallel environment.
 *
 *  $Id: test_pe.c,v 1.2 2010-11-02 17:51:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include "pe.h"
#include "tests.h"

int main(int argc, char ** argv) {

  int my_rank = 0;
  int my_size = 1;

  pe_init(argc, argv);

  info("Checking pe_init() (post-hoc)...");
  test_assert(1);
  info("ok\n");

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);

  info("Checking pe_rank() is correct...");
  test_assert(pe_rank() == my_rank);
  info("yes\n");

  info("Checking pe_size() is correct...");
  test_assert(pe_size() == my_size);
  info("yes\n");

  verbose("Checking verbose()\n");

  info("Verbose working ok...");
  test_assert(1);
  info("yes\n");

  info("About to do pe_finalise()...\n");
  pe_finalise();

  return 0;
}
