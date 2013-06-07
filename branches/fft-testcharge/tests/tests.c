/*****************************************************************************
 *
 *  tests.c
 *
 *  $Id: tests.c,v 1.4 2010-11-02 17:51:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "tests.h"

/*****************************************************************************
 *
 *  test_assert
 *
 *  Asimple assertion to control what happens in parallel.
 *
 *****************************************************************************/

void test_assert(const int lvalue) {

  int rank;

  if (lvalue) {
    /* ok */
  }
  else {
    /* Who has failed? */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[%d] ***************** Failed test assertion\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  return;
}
