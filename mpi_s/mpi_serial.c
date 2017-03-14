/******************************************************************************
 *
 *  mpi_serial
 *
 *  Library to replace MPI in serial.
 *
 *  From an idea appearing in, for example,  LAMMPS.
 *
 *    Point-to-point communications: will terminate.
 *    Collective communications: copy for basic datatypes
 *    Groups, Contexts, Comunicators: mostly no-operations
 *    Process Topologies: no operations
 *    Environmental inquiry: mostly operational
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use clock() only as a last resort in serial */

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#define omp_get_wtime() ((double) clock()*(1.0/CLOCKS_PER_SEC))
#define omp_get_wtick() (1.0/CLOCKS_PER_SEC)
#endif

#include "mpi.h"

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type);
static int mpi_sizeof(MPI_Datatype type);

static int mpi_initialised_flag_ = 0;
static int periods_[3];

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
 *  MPI_Bcast
 *
 *  No operation.
 *
 *****************************************************************************/

int MPI_Bcast(void * buffer, int count, MPI_Datatype datatype, int root,
	      MPI_Comm comm) {

  assert(mpi_initialised_flag_);
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

  return omp_get_wtime();
}

/*****************************************************************************
 *
 *  MPI_Wtick
 *
 *****************************************************************************/

double MPI_Wtick(void) {

  return omp_get_wtick();
}

/*****************************************************************************
 *
 *  MPI_Send
 *
 *****************************************************************************/

int MPI_Send(void * buf, int count, MPI_Datatype datatype, int dest,
	     int tag, MPI_Comm comm) {


  printf("MPI_Send should not be called in serial.\n");
  exit(0);

  return MPI_SUCCESS;
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
 *  MPI_Waitany
 *
 *****************************************************************************/

int MPI_Waitany(int count, MPI_Request requests[], int * index,
		MPI_Status * statuses) {

  return MPI_SUCCESS;
}


/*****************************************************************************
 *
 *  MPI_Probe
 *
 *****************************************************************************/

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status) {

  printf("MPI_Probe should not be called in serial\n");
  exit(0);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Sendrecv
 *
 *****************************************************************************/

int MPI_Sendrecv(void * sendbuf, int sendcount, MPI_Datatype sendtype,
		 int dest, int sendtag, void * recvbuf, int recvcount,
		 MPI_Datatype recvtype, int source, int recvtag,
		 MPI_Comm comm, MPI_Status * status) {

  printf("MPI_Sendrecv should not be called in serial\n");
  exit(0);

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

/****************************************************************************
 *
 *  MPI_Allgather
 *
 ****************************************************************************/

int MPI_Allgather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
		  void * recvbuf, int recvcount, MPI_Datatype recvtype,
		  MPI_Comm comm) {

  assert(mpi_initialised_flag_);
  assert(sendcount == recvcount);
  assert(sendtype == recvtype);
  mpi_copy(sendbuf, recvbuf, sendcount, sendtype);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Gather
 *
 ****************************************************************************/

int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
	       void * recvbuf, int recvcount, MPI_Datatype recvtype,
	       int root, MPI_Comm comm) {

  assert(mpi_initialised_flag_);
  assert(sendcount == recvcount);
  assert(sendtype == recvtype);
  
  mpi_copy(sendbuf, recvbuf, sendcount, sendtype);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Gatherv
 *
 *  The various assertions should be true in serial.
 *
 *****************************************************************************/

int MPI_Gatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype,
		void * recvbuf, const int * recvcounts, const int * displ,
		MPI_Datatype recvtype, int root, MPI_Comm comm) {

  assert(sendbuf);
  assert(recvbuf);
  assert(root == 0);
  assert(sendtype == recvtype);
  assert(sendcount == recvcounts[0]);

  mpi_copy((void *) sendbuf, recvbuf, sendcount, sendtype);

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
 *  MPI_Comm_dup
 *
 *  Just return the old one.
 *
 *****************************************************************************/

int MPI_Comm_dup(MPI_Comm oldcomm, MPI_Comm * newcomm) {

  assert(mpi_initialised_flag_);
  assert(oldcomm != MPI_COMM_NULL);

  *newcomm = oldcomm;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_indexed
 *
 *****************************************************************************/

int MPI_Type_indexed(int count, int * array_of_blocklengths,
		     int * array_of_displacements, MPI_Datatype oldtype,
		     MPI_Datatype * newtype) {

  *newtype = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_contiguous
 *
 *****************************************************************************/

int MPI_Type_contiguous(int count, MPI_Datatype old, MPI_Datatype * newtype) {

  *newtype = MPI_UNDEFINED;

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
 *  MPI_Type_struct
 *
 *****************************************************************************/

int MPI_Type_struct(int count, int * array_of_blocklengths,
		    MPI_Aint * array_of_displacements,
		    MPI_Datatype * array_of_types, MPI_Datatype * newtype) {

  *newtype = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Address
 *
 *  Please use MPI_Get_Address().
 *
 *****************************************************************************/

int MPI_Address(void * location, MPI_Aint * address) {

  *address = 0;

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

  int n;
  assert(mpi_initialised_flag_);
  assert(ndims <= 3);
  *newcomm = oldcomm;

  for (n = 0; n < ndims; n++) {
    periods_[n] = periods[n];
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_get
 *
 *  The state held for periods[] is only required for the tests.
 *
 *****************************************************************************/

int MPI_Cart_get(MPI_Comm comm, int maxdims, int * dims, int * periods,
		 int * coords) {

  int n;
  assert(mpi_initialised_flag_);

  for (n = 0; n < maxdims; n++) {
    dims[n] = 1;
    periods[n] = periods_[n];
    coords[n] = 0;
  }

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
 *  MPI_Cart_sub
 *
 *****************************************************************************/

int MPI_Cart_sub(MPI_Comm comm, int * remain_dims, MPI_Comm * new_comm) {

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);

  *new_comm = comm;

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

  int size = sizeof(int);

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
    printf("MPI_PACKED not implemented\n");
  default:
    printf("Unrecognised data type\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  return size;
}


/*****************************************************************************
 *
 *  MPI_Comm_set_errhandler
 *
 *****************************************************************************/

int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) {

  assert(mpi_initialised_flag_);
  assert(comm != MPI_COMM_NULL);
  assert(errhandler == MPI_ERRORS_ARE_FATAL);

  return MPI_SUCCESS;
}


#ifdef _DO_NOT_INCLUDE_MPI2_INTERFACE
/*
 * The following are removed from MPI3... and have an apprpriate
 * MPI2 replacement.
 *
 * MPI_Address           ->    MPI_Get_address
 * MPI_Type_hindexed     ->    MPI_Type_create_hindexed
 * MPI_Type_hvector      ->    MPI_Type_create_hvector
 * MPI_Type_struct       ->    MPI_Type_create_struct
 * MPI_Type_ub           ->    MPI_Type_get_extent
 * MPI_Type_lb           ->    MPI_Type_get_extent
 * MPI_LB                ->    MPI_Type_create_resized
 * MPI_UB                ->    MPI_Type_create_resized
 * MPI_Errhandler_create ->    MPI_Comm_create_errhandler
 * MPI_Errhandler_get    ->    MPI_Comm_get_errhandler
 * MPI_Errhandler_set    ->    MPI_Comm_set_errhandler
 * MPI_Handler_function  ->    MPI_Comm_errhandler_function
 * MPI_Keyval_create     ->    MPI_Comm_create_keyval
 * MPI_Keyval_free       ->    MPI_Comm_free_keyval
 * MPI_Dup_fn            ->    MPI_Comm_dup_fn
 * MPI_Null_copy_fn      ->    MPI_Comm_null_copy_fn
 * MPI_Null_delete_fn    ->    MPI_Comm_null_delete_fn
 * MPI_Copy_function     ->    MPI_Comm_copy_attr_function
 * COPY_FUNCTION         ->    COMM_COPY_ATTR_FN
 * MPI_Delete_function   ->    MPI_Comm_delete_attr_function
 * DELETE_FUNCTION       ->    COMM_DELETE_ATTR_FN
 * MPI_ATTR_DELETE       ->    MPI_Comm_delete_attr
 * MPI_Attr_get          ->    MPI_Comm_get_attr
 * MPI_Attr_put          ->    MPI_Comm_set_atrr
 */
#else

/*****************************************************************************
 *
 *  MPI_Get_address
 *
 *  Supercedes MPI_Address
 *
 *****************************************************************************/

int MPI_Get_address(const void * location, MPI_Aint * address) {

  *address = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_create_resized
 *
 *****************************************************************************/

int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
			    MPI_Datatype * newtype) {

  *newtype = oldtype;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_create_struct
 *
 *  Supercedes MPI_Type_struct()
 *
 *****************************************************************************/

int MPI_Type_create_struct(int count, int array_of_blocklengths[],
			   const MPI_Aint array_of_displacements[],
			   const MPI_Datatype array_of_types[],
			   MPI_Datatype * newtype) {

  *newtype = MPI_UNDEFINED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_get_extent
 *
 *****************************************************************************/

int MPI_Type_get_extent(MPI_Datatype dataype, MPI_Aint * lb,
			MPI_Aint * extent) {

  *lb = 0;
  *extent = -1;

  return MPI_SUCCESS;
}

#endif /* _DO_NOT_INCLUDE_MPI2_INTERFACE */
