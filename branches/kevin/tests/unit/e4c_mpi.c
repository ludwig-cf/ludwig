/*****************************************************************************
 *
 *  e4c_mpi.c
 *
 *  Exception implementation including MPI.
 *
 *****************************************************************************/

#include "e4c_mpi.h"

E4C_DEFINE_EXCEPTION(MPINullPointerException, "Null pointer.",
		     NullPointerException);
E4C_DEFINE_EXCEPTION(IOException, "IO Exception", RuntimeException);
E4C_DEFINE_EXCEPTION(MPIIOException, "MPI IO Exception", IOException);

E4C_DEFINE_EXCEPTION(TestException, "Test", TestException);
E4C_DEFINE_EXCEPTION(TestFailedException, "Test failed", TestException);
E4C_DEFINE_EXCEPTION(MPITestFailedException, "Test failed", TestException);

/*****************************************************************************
 *
 *  e4c_mpi_allreduce
 *
 *  Is any rank in this communicator showing an exception.
 *
 *****************************************************************************/

int e4c_mpi_allreduce(e4c_mpi_t e) {

  int ifail = 0;
  int ifail_local;

  ifail_local = (e.type != NULL);
  MPI_Allreduce(&ifail_local, &ifail, 1, MPI_INT, MPI_LOR, e.comm);

  return ifail;
}
