/*****************************************************************************
 *
 *  tests.c
 *
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#ifdef _MPI_
#include <mpi.h>
#endif

#include "tests.h"


/*****************************************************************************
 *
 *  test_assert
 *
 *  This is a generalisation of assert() from <assert.h> which
 *  controls what is happening in parallel.
 *
 *****************************************************************************/

void test_assert(const int lvalue) {

  if (lvalue) {
    /* ok */
  }
  else {
    /* Who has failed? */
#ifdef _MPI_
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[%d] Failed test assertion\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 0);
#else
    printf("\n** HALT!\n");
    printf("Failed test assertion\n");
    exit(0);
#endif
  }

  return;
}

/*****************************************************************************
 *
 *  test_barrier
 *
 *****************************************************************************/

void test_barrier() {

#ifdef _MPI_
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  return;
}
