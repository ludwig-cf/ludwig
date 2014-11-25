/*****************************************************************************
 *
 *  e4c_mpi.h
 *
 *  A number of simple extensions to Exception for C to make life
 *  a little easier under MPI.
 *
 *  Broadly, this follows three types of Exception:
 *    1. checked exceptions - which we should catch and recover from
 *    2. runtime - things we can catch but not usually recover from 
 *    3. errors - external events we should not try to catch
 *
 *  The programmer should arrange that exceptions prefixed MPI are
 *  thrown by all ranks in the relevant communicator, so they can
 *  be caught up the chain without further synchronisation.
 *
 *  The programmer must ensure that program logic avoids deadlock.
 *  There is no automatic way to do this.
 *
 *  A utility e4c_mpi_err(comm) is supplied to help to identify
 *  any failure within a communicator.
 *
 *****************************************************************************/

#ifndef E4C_MPI_H
#define E4C_MPI_H

#include "mpi.h"
#include "e4c_lite.h"

/* A facade merely to provide information to the reader. */
#define throws(MPIException, ...)

/* Runtime exceptions */

E4C_DECLARE_EXCEPTION(MPINullPointerException);
E4C_DECLARE_EXCEPTION(IOException);
E4C_DECLARE_EXCEPTION(MPIIOException);

/* Checked exceptions */

E4C_DECLARE_EXCEPTION(TestException);
E4C_DECLARE_EXCEPTION(TestFailedException);
E4C_DECLARE_EXCEPTION(MPITestFailedException);

typedef struct e4c_mpi_s e4c_mpi_t;

struct e4c_mpi_s {
  MPI_Comm comm;
  const struct e4c_exception_type * type;
};

#define e4c_mpi_init(e, comm) do {e.comm = comm; e.type = NULL;} while (0)
#define e4c_mpi_exception(e) do { e.type = e4c.err.type; } while(0)

int e4c_mpi_allreduce(e4c_mpi_t e);

#endif
