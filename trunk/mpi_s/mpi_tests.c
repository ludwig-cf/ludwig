/*****************************************************************************
 *
 *  mpi_tests
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include "mpi.h"

static MPI_Comm comm_ = MPI_COMM_WORLD;
static int test_mpi_comm_rank(void);
static int test_mpi_comm_size(void);
static int test_mpi_allreduce(void);

int main (int argc, char ** argv) {

  int ireturn;

  printf("Running mpi_s tests...\n");

  ireturn = MPI_Init(&argc, &argv);
  assert (ireturn == MPI_SUCCESS);

  ireturn = test_mpi_comm_rank();
  ireturn = test_mpi_comm_size();
  ireturn = test_mpi_allreduce();

  ireturn = MPI_Finalize();
  assert(ireturn == MPI_SUCCESS);

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_comm_rank
 *
 *****************************************************************************/

int test_mpi_comm_rank() {

  int rank = MPI_PROC_NULL;
  int ireturn;

  ireturn = MPI_Comm_rank(comm_, &rank);
  assert(ireturn == MPI_SUCCESS);
  assert(rank == 0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_comm_size
 *
 *****************************************************************************/

int test_mpi_comm_size() {

  int size = 0;
  int ireturn;

  ireturn = MPI_Comm_size(comm_, &size);
  assert(ireturn == MPI_SUCCESS);
  assert(size == 1);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_allreduce
 *
 *****************************************************************************/

int test_mpi_allreduce() {

  int ireturn;
  double dsend, drecv;
  int isend[3], irecv[3];

  dsend = 1.0; drecv = 0.0;
  ireturn = MPI_Allreduce(&dsend, &drecv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(dsend == 1.0);
  assert(drecv == dsend);

  isend[0] = -1;
  isend[1] = 0;
  isend[2] = +1;

  ireturn = MPI_Allreduce(isend, irecv, 3, MPI_INT, MPI_SUM, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(isend[0] == -1);
  assert(isend[1] == 0);
  assert(isend[2] == +1);
  assert(irecv[0] == -1);
  assert(irecv[1] == 0);
  assert(irecv[2] == +1);

  return MPI_SUCCESS;
}
