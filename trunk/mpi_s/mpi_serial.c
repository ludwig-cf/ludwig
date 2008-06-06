/******************************************************************************
 *
 *  mpi_serial
 *
 *  Library to replace MPI in serial.
 *
 *  From an idea appearing in, for example,  LAMMPS.
 *
 *  Current status:
 *    Point-to-point communications: will terminate.
 *    Collective communications: copy for basic datatypes
 *    Groups, Contexts, Comunicators: mostly no-operations
 *    Process Topologies: no operations
 *    Environmental inquiry: mostly operational
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mpi.h"

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type);
static int mpi_sizeof(MPI_Datatype type);

static int mpi_initialised_flag_ = 0;

/*****************************************************************************
 *
 *  MPI_Barrier
 *
 *****************************************************************************/

int MPI_Barrier(MPI_Comm comm) {

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Init
 *
 *****************************************************************************/

int MPI_Init(int * argc, char *** argv) {

  mpi_initialised_flag_ = 1;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Initialized
 *
 *****************************************************************************/

int MPI_Initialized(int * flag) {

  *flag = mpi_initialised_flag_;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Finalize
 *
 *****************************************************************************/

int MPI_Finalize(void) {

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_rank
 *
 *****************************************************************************/

int MPI_Comm_rank(MPI_Comm comm, int * rank) {

  *rank = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_size
 *
 *****************************************************************************/

int MPI_Comm_size(MPI_Comm comm, int * size) {

  *size = 1;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Abort
 *
 *****************************************************************************/

int MPI_Abort(MPI_Comm comm, int code) {

  exit(code);
  return MPI_SUCCESS; /* ha! */
}

/*****************************************************************************
 *
 *  MPI_Wtime
 *
 *  The number of seconds since the start of time(!)
 *
 *****************************************************************************/

double MPI_Wtime(void) {

  return ((double) clock() / CLOCKS_PER_SEC);
}

/*****************************************************************************
 *
 *  MPI_Wtick
 *
 *****************************************************************************/

double MPI_Wtick(void) {

  return (double) (1.0/ CLOCKS_PER_SEC);
}

/*****************************************************************************
 *
 *  MPI_Recv
 *
 *****************************************************************************/

int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source,
	     int tag, MPI_Comm comm, MPI_Status * status) {

  printf("MPI_Recv should not be called in serial.\n");
  exit(0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Irecv
 *
 *****************************************************************************/

int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source,
	     int tag, MPI_Comm comm, MPI_Request * request) {

  printf("MPI_Irecv should not be called in serial.\n");
  exit(0);

  return MPI_SUCCESS;
}


/*****************************************************************************
 *
 *  MPI_Ssend
 *
 *****************************************************************************/

int MPI_Ssend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm) {

  printf("MPI_Ssend should not be called in serial\n");
  exit(0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Isend
 *
 *****************************************************************************/

int MPI_Isend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm, MPI_Request * request) {

  printf("MPI_Isend should not be called in serial\n");
  exit(0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Issend
 *
 *****************************************************************************/

int MPI_Issend(void * buf, int count, MPI_Datatype datatype, int dest,
	       int tag, MPI_Comm comm, MPI_Request * request) {

  printf("MPI_Issend should not be called in serial\n");
  exit(0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Waitall
 *
 *****************************************************************************/

int MPI_Waitall(int count, MPI_Request * requests, MPI_Status * statuses) {

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Reduce
 *
 *****************************************************************************/

int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
	       MPI_Op op, int root, MPI_Comm comm) {

  /* mpi_check_collective(op);*/
  mpi_copy(sendbuf, recvbuf, count, type);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Allreduce
 *
 *****************************************************************************/

int MPI_Allreduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
		  MPI_Op op, MPI_Comm comm) {

  mpi_copy(sendbuf, recvbuf, count, type);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_split
 *
 *  Return the original communicator as the new communicator.
 *
 *****************************************************************************/

int MPI_Comm_split(MPI_Comm comm, int colour, int key, MPI_Comm * newcomm) {

  *newcomm = comm;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_free
 *
 *  No operation.
 *
 *****************************************************************************/

int MPI_Comm_free(MPI_Comm * comm) {

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_contiguous
 *
 *****************************************************************************/

int MPI_Type_contiguous(int count, MPI_Datatype old, MPI_Datatype * new) {

  *new = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_commit
 *
 *****************************************************************************/

int MPI_Type_commit(MPI_Datatype * type) {

  /* Flag this as undefined at the moment */
  *type = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_free
 *
 *****************************************************************************/

int MPI_Type_free(MPI_Datatype * type) {

  *type = MPI_DATATYPE_NULL;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_vector
 *
 *****************************************************************************/

int MPI_Type_vector(int count, int blocklength, int stride,
		    MPI_Datatype oldtype, MPI_Datatype * newtype) {

  *newtype = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Errhandler_set
 *
 *****************************************************************************/

int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler) {

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);
  assert(errhandler == MPI_ERRORS_ARE_FATAL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_create
 *
 *****************************************************************************/

int MPI_Cart_create(MPI_Comm oldcomm, int ndims, int * dims, int * periods,
		    int reorder, MPI_Comm * newcomm) {

  assert(mpi_initialised_flag_);
  *newcomm = oldcomm;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 * MPI_Cart_coords
 *
 *****************************************************************************/

int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int * coords) {

  int d;

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);
  assert(rank == 0);

  for (d = 0; d < maxdims; d++) {
    coords[d] = 0;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_rank
 *
 *  Set the Cartesian rank to zero.
 *
 *****************************************************************************/

int MPI_Cart_rank(MPI_Comm comm, int * coords, int * rank) {

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);
  *rank = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_shift
 *
 *  No attempt is made to deal with non-periodic boundaries.
 *
 *****************************************************************************/

int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source,
		   int * rank_dest) {

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);

  *rank_source = 0;
  *rank_dest = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Dims_create
 *
 *****************************************************************************/

int MPI_Dims_create(int nnodes, int ndims, int * dims) {

  int d;

  assert(mpi_initialised_flag_);
  assert(nnodes == 1);
  assert(ndims > 0);

  for (d = 0; d < ndims; d++) {
    dims[d] = 1;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  mpi_copy
 *
 *****************************************************************************/

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type) {
 
  int sizeof_datatype = mpi_sizeof(type);

  memcpy(recv, send, count*sizeof_datatype);

  return;
}

/*****************************************************************************
 *
 *  mpi_sizeof
 *
 *****************************************************************************/

static int mpi_sizeof(MPI_Datatype type) {

  int size;

  switch (type) {
  case MPI_CHAR:
    size = sizeof(char);
    break;
  case MPI_SHORT:
    size = sizeof(short int);
    break;
  case MPI_INT:
    size = sizeof(int);
    break;
  case MPI_LONG:
    size = sizeof(long int);
    break;
  case MPI_UNSIGNED_CHAR:
    size = sizeof(unsigned char);
    break;
  case MPI_UNSIGNED_SHORT:
    size = sizeof(unsigned short int);
    break;
  case MPI_UNSIGNED:
    size = sizeof(unsigned int);
    break;
  case MPI_UNSIGNED_LONG:
    size = sizeof(unsigned long int);
    break;
  case MPI_FLOAT:
    size = sizeof(float);
    break;
  case MPI_DOUBLE:
    size = sizeof(double);
    break;
  case MPI_LONG_DOUBLE:
    size = sizeof(double);
    break;
  case MPI_BYTE:
    size = sizeof(char);
    break;
  case MPI_PACKED:
    /* mpi_error("MPI_PACKED not implemented\n");*/
    break;
  default:
    ;
    /* mpi_error("Unrecognised data type\n");*/
  }

  return size;
}

