/*****************************************************************************
 *
 *  e4c_mpi.c
 *
 *  Exception implementation including MPI.
 *
 *****************************************************************************/

#include "e4c_mpi.h"

E4C_DEFINE_EXCEPTION(MPINullPointerException, "Null pointer.",
		     RuntimeException);

E4C_DEFINE_EXCEPTION(TestException, "Test", TestException);
E4C_DEFINE_EXCEPTION(TestFailedException, "Test failed", TestException);
E4C_DEFINE_EXCEPTION(MPITestFailedException, "Test failed", TestException);

/*****************************************************************************
 *
 *  e4c_mpi_err
 *
 *  Is any rank in this communicator showing an exception at any
 *  point in the past?
 *
 *****************************************************************************/

int e4c_mpi_err(MPI_Comm comm) {

  int ifail = 0;
  int ifail_local;

  ifail_local = (e4c.err.type != NULL);
  MPI_Allreduce(&ifail_local, &ifail, 1, MPI_INT, MPI_LOR, comm);

  return ifail;
}
