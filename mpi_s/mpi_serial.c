/******************************************************************************
 *
 *  mpi_serial
 *
 *  Library to replace MPI in serial.
 *
 *  From an idea appearing in, for example, LAMMPS, and elsewhere.
 *
 *    Point-to-point communications: may terminate.
 *    Collective communications: copy for basic datatypes
 *    Groups, Contexts, Comunicators: mostly no-operations
 *    Process Topologies: no operations
 *    Environmental inquiry: mostly operational
 *    MPI-IO: basic operations
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use clock() only as a last resort in serial (no threads) */

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#define omp_get_wtime() ((double) clock()*(1.0/CLOCKS_PER_SEC))
#define omp_get_wtick() (1.0/CLOCKS_PER_SEC)
#endif

#include "mpi.h"

/* Internal state */

#define MAX_CART_COMM  128
#define MAX_USER_DT    128
#define MAX_USER_FILE  16

/* We are not going to deal with all possible data types; encode
 * what we have ... */
enum dt_flavour {DT_NOT_IMPLEMENTED = 0, DT_CONTIGUOUS, DT_VECTOR, DT_STRUCT,
                 DT_SUBARRAY};

typedef struct internal_data_type_s data_t;
typedef struct internal_file_view_s file_t;

struct internal_data_type_s {
  MPI_Datatype handle;       /* User space handle [in suitable range] */
  int          bytes;        /* sizeof */
  int          commit;       /* Commited? */
  int          flavour;      /* Contiguous types only at present */
};

struct internal_file_view_s {
  FILE * fp;                    /* file pointer */
  MPI_Offset   disp;            /* e.g., from MPI_File_set_view() */
  MPI_Datatype etype;
  MPI_Datatype filetype;
  char datarep[MPI_MAX_DATAREP_STRING];
};

typedef struct mpi_info_s mpi_info_t;

struct mpi_info_s {
  int initialised;               /* MPI initialised */
  int ncart;                     /* Number of Cartesian communicators */
  int period[MAX_CART_COMM][3];  /* Periodic Cartesisan per communicator */
  int ndatatype;                 /* Current number of data types */
  data_t dt[MAX_USER_DT];        /* Internal information per data type */
  int ndatatypelast;             /* Current free list extent */
  int dtfreelist[MAX_USER_DT];   /* Free list */

  file_t filelist[MAX_USER_FILE]; /* MPI_File information for open files */
};

static mpi_info_t * mpi_info = NULL;

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type);
static int mpi_sizeof(MPI_Datatype type);
static int mpi_sizeof_user(MPI_Datatype handle);
static int mpi_is_valid_comm(MPI_Comm comm);
static int mpi_data_type_add(mpi_info_t * ctxt, const data_t * dt,
			     MPI_Datatype * newtype);
static int mpi_data_type_free(mpi_info_t * ctxt, MPI_Datatype * handle);
static int mpi_data_type_handle(mpi_info_t * ctxt, MPI_Datatype handle);

static MPI_File mpi_file_handle_retain(mpi_info_t * ctxt, FILE * fp);
static FILE *   mpi_file_handle_release(mpi_info_t * ctxt, MPI_File handle);
static FILE *   mpi_file_handle_to_fp(mpi_info_t * info, MPI_File handle);

/*****************************************************************************
 *
 *  MPI_Barrier
 *
 *****************************************************************************/

int MPI_Barrier(MPI_Comm comm) {

  assert(mpi_is_valid_comm(comm));

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

  assert(mpi_info->initialised);
  assert(buffer);
  assert(count > 0);
  assert(mpi_is_valid_comm(comm));

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Init
 *
 *****************************************************************************/

int MPI_Init(int * argc, char *** argv) {

  assert(argc);
  assert(argv);

  mpi_info = (mpi_info_t *) calloc(1, sizeof(mpi_info_t));
  assert(mpi_info);

  mpi_info->initialised = 1;

  /* User data type handles: reserve dt[0].handle = 0 for MPI_DATATYPE_NULL */
  /* Otherwise, user data type handles are indexed 1, 2, 3, ... */
  /* Initialise the free list. */

  assert(MPI_DATATYPE_NULL == 0);

  for (int n = 0; n < MAX_USER_DT; n++) {
    data_t null = {.handle = 0, .bytes = 0, .commit = 0, .flavour = 0};
    mpi_info->dt[n] = null;
    mpi_info->dtfreelist[n] = n;
  }
  mpi_info->ndatatype = 0;
  mpi_info->ndatatypelast = 0;

  for (int ih = 0; ih < MAX_USER_FILE; ih++) {
    mpi_info->filelist[ih].fp    = NULL;
    mpi_info->filelist[ih].disp  = 0;
    mpi_info->filelist[ih].etype = MPI_BYTE;
    mpi_info->filelist[ih].filetype = MPI_BYTE;
    strncpy(mpi_info->filelist[ih].datarep, "native", MPI_MAX_DATAREP_STRING);
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Init_thread
 *
 *****************************************************************************/

int MPI_Init_thread(int * argc, char *** argv, int required, int * provided) {

  assert(argc);
  assert(argv);
  assert(MPI_THREAD_SINGLE <= required && required <= MPI_THREAD_MULTIPLE);
  assert(provided);

  MPI_Init(argc, argv);

  /* We are going to say that MPI_THREAD_SERIALIZED is available */
  /* Not MPI_THREAD_MULTIPLE */

  *provided = MPI_THREAD_SERIALIZED;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Initialized
 *
 *****************************************************************************/

int MPI_Initialized(int * flag) {

  assert(flag);

  *flag = (mpi_info != NULL); /* A sufficient condition */

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Finalize
 *
 *****************************************************************************/

int MPI_Finalize(void) {

  assert(mpi_info);
  assert(mpi_info->ndatatype == 0); /* All released */

  free(mpi_info);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_group
 *
 *****************************************************************************/

int MPI_Comm_group(MPI_Comm comm, MPI_Group * group) {

  assert(mpi_is_valid_comm(comm));
  assert(group);

  *group = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_rank
 *
 *****************************************************************************/

int MPI_Comm_rank(MPI_Comm comm, int * rank) {

  assert(mpi_is_valid_comm(comm));
  assert(rank);

  *rank = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_size
 *
 *****************************************************************************/

int MPI_Comm_size(MPI_Comm comm, int * size) {

  assert(mpi_is_valid_comm(comm));
  assert(size);

  *size = 1;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_compare
 *
 *  This is not quite right at the moment, as Cartesian communicators
 *  and non-Cartesian communictors need to treated on an equal basis.
 *  Specifically MPI_Comm_split() does note return a unique value.
 *
 *****************************************************************************/

int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int * result) {

  assert(result);

  *result = MPI_UNEQUAL;
  if (mpi_is_valid_comm(comm1) && mpi_is_valid_comm(comm2)) {
    *result = MPI_CONGRUENT;
    if (comm1 == comm2) *result = MPI_IDENT;
  }

  return 0;
}

/*****************************************************************************
 *
 *  MPI_Abort
 *
 *****************************************************************************/

int MPI_Abort(MPI_Comm comm, int code) {

  int is_valid;

  is_valid = 1 - mpi_is_valid_comm(comm);

  exit(code + is_valid);
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

  assert(buf);

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

  assert(buf);

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

  assert(buf);
  assert(request);

  /* Could assert tag is ok */
  *request = tag;

  return MPI_SUCCESS;
}


/*****************************************************************************
 *
 *  MPI_Ssend
 *
 *****************************************************************************/

int MPI_Ssend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm) {

  assert(buf);

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

  assert(buf);
  assert(request);

  /* Could assert tag is ok */
  *request = tag;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Issend
 *
 *****************************************************************************/

int MPI_Issend(void * buf, int count, MPI_Datatype datatype, int dest,
	       int tag, MPI_Comm comm, MPI_Request * request) {

  assert(buf);
  assert(count >= 0);
  assert(dest == 0);
  assert(mpi_is_valid_comm(comm));
  assert(request);

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

  assert(count >= 0);
  assert(requests);

  return MPI_SUCCESS;
}


/*****************************************************************************
 *
 *  MPI_Waitany
 *
 *****************************************************************************/

int MPI_Waitany(int count, MPI_Request requests[], int * index,
		MPI_Status * status) {

  assert(count >= 0);
  assert(requests);
  assert(index);

  *index = MPI_UNDEFINED;

  for (int ireq = 0; ireq < count; ireq++) {
    if (requests[ireq] != MPI_REQUEST_NULL) {
      *index = ireq;
      requests[ireq] = MPI_REQUEST_NULL;
      if (status) {
	status->MPI_SOURCE = 0;
	status->MPI_TAG = requests[ireq];
      }
      break;
    }
  }

  return MPI_SUCCESS;
}


/*****************************************************************************
 *
 *  MPI_Probe
 *
 *****************************************************************************/

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status) {

  assert(source == 0);
  assert(mpi_is_valid_comm(comm));

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

  assert(sendbuf);
  assert(dest == source);
  assert(recvbuf);
  assert(recvcount == sendcount);

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

  assert(sendbuf);
  assert(recvbuf);
  assert(count >= 0);
  assert(root == 0);
  assert(mpi_is_valid_comm(comm));

  assert(op != MPI_OP_NULL);

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

  assert(mpi_info);
  assert(sendbuf);
  assert(recvbuf);
  assert(sendcount == recvcount);
  assert(sendtype == recvtype);
  assert(mpi_is_valid_comm(comm));

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

  assert(mpi_info);
  assert(sendbuf);
  assert(recvbuf);
  assert(sendcount == recvcount);
  assert(sendtype == recvtype);
  assert(mpi_is_valid_comm(comm));
  
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

  assert(sendbuf);
  assert(recvbuf);
  assert(count >= 1);
  assert(mpi_is_valid_comm(comm));

  if (sendbuf != MPI_IN_PLACE) {
    mpi_copy(sendbuf, recvbuf, count, type);
  }

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

  assert(mpi_is_valid_comm(comm));
  assert(newcomm);

  /* Allow that a split Cartesian communicator is different */
  /* See MPI_Comm_compare() */
  *newcomm = MPI_COMM_WORLD;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_split_type
 *
 *****************************************************************************/

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info,
			MPI_Comm * newcomm) {

  assert(mpi_is_valid_comm(comm));
  assert(newcomm);
  assert(split_type == MPI_COMM_TYPE_SHARED);

  *newcomm = comm;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Comm_free
 *
 *****************************************************************************/

int MPI_Comm_free(MPI_Comm * comm) {

  /* Mark Cartesian communicators as free */

  if (*comm > MPI_COMM_SELF) {
    mpi_info->ncart -= 1;
  }

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

  assert(mpi_info);
  assert(mpi_is_valid_comm(oldcomm));
  assert(newcomm);

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

  assert(count > 0);
  assert(array_of_blocklengths);
  assert(array_of_displacements);
  assert(newtype);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = 0;
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Can't do displacements at moment */

    mpi_data_type_add(mpi_info, &dt, newtype);
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_contiguous
 *
 *****************************************************************************/

int MPI_Type_contiguous(int count, MPI_Datatype old, MPI_Datatype * newtype) {

  assert(count > 0);
  assert(newtype);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = mpi_sizeof(old)*count;  /* contiguous */
    dt.commit  = 0;
    dt.flavour = DT_CONTIGUOUS;

    mpi_data_type_add(mpi_info, &dt, newtype);
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_commit
 *
 *****************************************************************************/

int MPI_Type_commit(MPI_Datatype * type) {

  assert(type);

  int handle = *type;

  if (handle < 0) {
    printf("MPI_Type_commit: Attempt to commit intrinsic type\n");
  }
  if (handle == 0) {
    printf("MPI_Type_commit: Attempt to commit null data type\n");
  }
  if (handle > mpi_info->ndatatypelast) {
    printf("MPI_Type_commit: unrecognised handle %d\n", handle);
  }

  assert(mpi_info->dt[handle].handle == handle);
  mpi_info->dt[handle].commit = 1;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_free
 *
 *****************************************************************************/

int MPI_Type_free(MPI_Datatype * type) {

  assert(type);

  mpi_data_type_free(mpi_info, type);
  assert(*type == MPI_DATATYPE_NULL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_vector
 *
 *****************************************************************************/

int MPI_Type_vector(int count, int blocklength, int stride,
		    MPI_Datatype oldtype, MPI_Datatype * newtype) {

  assert(count > 0);
  assert(blocklength >= 0);
  assert(newtype);

  {
    data_t dt = {0};

    dt.handle = MPI_DATATYPE_NULL;
    dt.bytes  = 0;
    dt.commit = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Can't do strided copy */

    mpi_data_type_add(mpi_info, &dt, newtype);
  }

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
  int icart;

  assert(mpi_info);
  assert(ndims <= 3);
  assert(newcomm);

  mpi_info->ncart += 1;
  icart = MPI_COMM_SELF + mpi_info->ncart;
  assert(icart < MAX_CART_COMM);

  *newcomm = icart;

  /* Record periodity */

  for (n = 0; n < ndims; n++) {
    mpi_info->period[icart][n] = periods[n];
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

  assert(mpi_info);
  assert(mpi_is_valid_comm(comm));
  assert(dims);
  assert(periods);
  assert(coords);

  for (n = 0; n < maxdims; n++) {
    dims[n] = 1;
    periods[n] = mpi_info->period[comm][n];
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

  assert(mpi_info);
  assert(comm != MPI_COMM_NULL);
  assert(rank == 0);
  assert(coords);

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

  assert(mpi_info);
  assert(mpi_is_valid_comm(comm));
  assert(coords);
  assert(rank);

  *rank = 0;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_shift
 *
 *****************************************************************************/

int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source,
		   int * rank_dest) {

  assert(mpi_info);
  assert(comm != MPI_COMM_NULL);
  assert(rank_source);
  assert(rank_dest);

  *rank_source = 0;
  *rank_dest = 0;

  /* Non periodic directions */
  if (disp != 0 && mpi_info->period[comm][direction] != 1) {
    *rank_source = MPI_PROC_NULL;
    *rank_dest   = MPI_PROC_NULL;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Cart_sub
 *
 *****************************************************************************/

int MPI_Cart_sub(MPI_Comm comm, int * remain_dims, MPI_Comm * new_comm) {

  assert(mpi_info);
  assert(mpi_is_valid_comm(comm));
  assert(remain_dims);
  assert(new_comm);

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

  assert(mpi_info);
  assert(nnodes == 1);
  assert(ndims > 0);
  assert(dims);

  for (d = 0; d < ndims; d++) {
    dims[d] = 1;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Op_create
 *
 *****************************************************************************/

int MPI_Op_create(MPI_User_function * function, int commute, MPI_Op * op) {

  /* Never actually use function, so don't really care what's here... */

  assert(function);

  *op = MPI_SUM;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Op_free
 *
 *****************************************************************************/

int MPI_Op_free(MPI_Op * op) {

  assert(op);

  *op = MPI_OP_NULL;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  mpi_copy
 *
 *****************************************************************************/

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type) {
 
  size_t sizeof_datatype = mpi_sizeof(type);

  assert(send);
  assert(recv);
  assert(count >= 0);

  memcpy(recv, send, sizeof_datatype*count);

  return;
}

/*****************************************************************************
 *
 *  mpi_sizeof
 *
 *****************************************************************************/

static int mpi_sizeof(MPI_Datatype type) {

  int size = -1;

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
    assert(sizeof(double) == sizeof(long double));
    break;
  case MPI_BYTE:
    size = sizeof(char);
    break;
  case MPI_INT32_T:
    size = sizeof(int32_t);
    break;
  case MPI_INT64_T:
    size = sizeof(int64_t);
    break;
  case MPI_PACKED:
    printf("MPI_PACKED not implemented\n");
  default:
    /* Try user type */
    size = mpi_sizeof_user(type);
  }

  assert(size != -1);
  return size;
}

/*****************************************************************************
 *
 *  mpi_sizeof_user
 *
 *  For user defined data types.
 *
 *****************************************************************************/

static int mpi_sizeof_user(MPI_Datatype handle) {

  int sz    = -1;
  int index = mpi_data_type_handle(mpi_info, handle);

  assert(index >= MPI_COMM_NULL); /* not intrinsic */

  if (index == MPI_DATATYPE_NULL) {
    printf("mpi_sizeof_dt: NULL data type %d\n", handle);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  else {
    sz = mpi_info->dt[index].bytes;
  }

  return sz;
}

/*****************************************************************************
 *
 *  MPI_Comm_set_errhandler
 *
 *****************************************************************************/

int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) {

  assert(mpi_info);
  assert(mpi_is_valid_comm(comm));
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

  assert(location);
  assert(address);

  *address = (MPI_Aint) location;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Group_translate_ranks
 *
 *****************************************************************************/

int MPI_Group_translate_ranks(MPI_Group grp1, int n, const int * ranks1,
			      MPI_Group grp2, int * ranks2) {
  assert(ranks1);
  assert(ranks2);

  memcpy(ranks2, ranks1, n*sizeof(int));

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_create_resized
 *
 *****************************************************************************/

int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
			    MPI_Datatype * newtype) {

  assert(newtype);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = extent;
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Should be  old.flavour */

    mpi_data_type_add(mpi_info, &dt, newtype);
  }

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

  assert(count > 0);
  assert(array_of_blocklengths);
  assert(array_of_displacements);
  assert(array_of_types);
  assert(newtype);


  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = 0;
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* General copy not available yet */

    /* Extent */
    /* C must maintain order so we should be able to write... */
    dt.bytes = (array_of_displacements[count-1] - array_of_displacements[0])
      + mpi_sizeof(array_of_types[count-1]);

    mpi_data_type_add(mpi_info, &dt, newtype);
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_get_extent
 *
 *****************************************************************************/

int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint * lb,
			MPI_Aint * extent) {

  int handle = MPI_DATATYPE_NULL;

  assert(lb);
  assert(extent);

  if (datatype < 0) {
    /* intrinsic allowed? Why? */
    assert(0);
    *extent = mpi_sizeof(datatype);
  }

  handle = mpi_data_type_handle(mpi_info, datatype);

  if (handle == MPI_DATATYPE_NULL) {
    printf("MPI_Type_get_Extent: null handle\n");
  }

  *lb = 0; /* Always, at the moment */
  *extent = mpi_info->dt[handle].bytes;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_size
 *
 *****************************************************************************/

int MPI_Type_size(MPI_Datatype datatype, int * sz) {

  assert(sz);

  *sz = mpi_sizeof(datatype);

  return 0;
}

/*****************************************************************************
 *
 *  MPI_File_open
 *
 *****************************************************************************/

int MPI_File_open(MPI_Comm comm, const char * filename, int amode,
		  MPI_Info info, MPI_File * fh) {

  FILE * fp = NULL;
  const char * fdmode = NULL;

  assert(mpi_is_valid_comm(comm));
  assert(filename);
  assert(fh);

  /* Exactly one of RDONLY, WRONLY, or RDWR must be present. */
  /* RDONLY => no CREATE or EXCL. */
  /* RDWR   => no SEQUENTIAL */

  {
    int have_rdonly = (amode & MPI_MODE_RDONLY) ? 1 : 0;
    int have_wronly = (amode & MPI_MODE_WRONLY) ? 2 : 0;
    int have_rdwr   = (amode & MPI_MODE_RDWR)   ? 4 : 0;

    int have_create = (amode & MPI_MODE_CREATE);
    int have_excl   = (amode & MPI_MODE_EXCL);
    int have_append = (amode & MPI_MODE_APPEND);

    switch (have_rdonly + have_wronly + have_rdwr) {
    case (1):
      /* Read only */
      fdmode = "r";
      if (have_create) printf("No create with RDONLY\n");
      if (have_excl)   printf("No excl   with RDONLY\n");
      break;
    case (2):
      /* Write only  */
      fdmode = "w";
      if (have_append) fdmode = "a";
      break;
    case (4):
      /* Read write */
      fdmode = "r+";
      if (have_append) fdmode = "a+";
      break;
    default:
      printf("Please specify exactly one of MPI_MODE_RDONLY, MPI_MODE_WRONLY, "
	     "or MPI_MODE_RDWR\n");
    }
  }

  /* Open the file and record the handle */

  fp = fopen(filename, fdmode);

  if (fp == NULL) {
    printf("MPI_File_open: attempt to open %s mode %s failed\n", filename,
	   fdmode);
    return MPI_ERR_NO_SUCH_FILE;
  }

  *fh = mpi_file_handle_retain(mpi_info, fp);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_close
 *
 *****************************************************************************/

int MPI_File_close(MPI_File * fh) {

  FILE * fp = NULL;

  assert(fh);

  fp = mpi_file_handle_release(mpi_info, *fh);

  if (fp == NULL) {
    printf("MPI_File_close: invalid file handle\n");
    return MPI_ERR_FILE;
  }
  else {
    fclose(fp);
    *fh = MPI_FILE_NULL;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_delete
 *
 *****************************************************************************/

int MPI_File_delete(const char * filename, MPI_Info info) {

  assert(filename);

  remove(filename);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Type_create_subarray
 *
 *****************************************************************************/

int MPI_Type_create_subarray(int ndims, const int * array_of_sizes,
			     const int * array_of_subsizes,
			     const int * array_of_starts,
			     int order,
			     MPI_Datatype oldtype,
			     MPI_Datatype * newtype) {

  int nelements = 0;

  assert(ndims == 2 || ndims == 3); /* We accept this is not general */
  assert(array_of_sizes);
  assert(array_of_subsizes);
  assert(array_of_starts);
  assert(order == MPI_ORDER_C || order == MPI_ORDER_FORTRAN);
  assert(newtype);

  /* Assume this is a contiguous block of elements of oldtype */
  nelements = array_of_sizes[0]*array_of_sizes[1];
  if (ndims == 3) nelements *= array_of_sizes[2];

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = mpi_sizeof(oldtype)*nelements;
    dt.commit  = 0;
    dt.flavour = DT_SUBARRAY;

    mpi_data_type_add(mpi_info, &dt, newtype);
  }
  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_get_view
 *
 *****************************************************************************/

int MPI_File_get_view(MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype,
		      MPI_Datatype * filetype, char * datarep) {

  FILE * fp = NULL;

  assert(disp);
  assert(etype);
  assert(filetype);
  assert(datarep);
  assert(mpi_info);

  fp = mpi_file_handle_to_fp(mpi_info, fh);

  if (fp == NULL) {
    printf("MPI_File_get_view: invalid file handle\n");
    exit(0);
  }
  else {
    file_t * file = &mpi_info->filelist[fh];
    *disp = file->disp;
    *etype = file->etype;
    *filetype = file->filetype;
    strncpy(datarep, file->datarep, MPI_MAX_DATAREP_STRING-1);
  }

  return MPI_SUCCESS;
}
/*****************************************************************************
 *
 *  MPI_File_set_view
 *
 *****************************************************************************/

int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
		      MPI_Datatype filetype, const char * datarep,
		      MPI_Info info) {

  FILE * fp = NULL;

  assert(datarep);
  assert(mpi_info);

  fp = mpi_file_handle_to_fp(mpi_info, fh);

  if (fp == NULL) {
    printf("MPI_File_set_view: invalid file handle\n");
    exit(0);
  }
  else {
    file_t * file = &mpi_info->filelist[fh];
    file->disp = disp;
    file->etype = etype;
    file->filetype = filetype;
    /* Could demand "native" ... */
    strncpy(file->datarep, datarep, MPI_MAX_DATAREP_STRING-1);
    /* info is currently discarded */
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_read_all
 *
 *****************************************************************************/

int MPI_File_read_all(MPI_File fh, void * buf, int count,
		      MPI_Datatype datatype, MPI_Status * status) {
  FILE * fp = NULL;

  assert(buf);

  fp = mpi_file_handle_to_fp(mpi_info, fh);

  if (fp == NULL) {
    printf("MPI_File_read_all: invalid_file handle\n");
    exit(0);
  }
  else {

    /* Translate to a simple fread() */

    size_t size   = mpi_sizeof(datatype);
    size_t nitems = count;
    size_t nr = fread(buf, size, nitems, fp);

    if (nr != nitems) {
      printf("MPI_File_read_all(): incorrect number of items in fread()\n");
    }

    if (ferror(fp)) {
      perror("perror: ");
      printf("MPI_File_read_all() file operation failed\n");
      exit(0);
    }
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_write_all
 *
 *****************************************************************************/

int MPI_File_write_all(MPI_File fh, const void * buf, int count,
		       MPI_Datatype datatype, MPI_Status * status) {

  FILE * fp = NULL;

  assert(buf);

  fp = mpi_file_handle_to_fp(mpi_info, fh);

  if (fp == NULL) {
    printf("MPI_File_write_all: invalid_file handle");
    exit(0);
  }
  else {

    /* Translate to a simple fwrite() */

    size_t size   = mpi_sizeof(datatype);
    size_t nitems = count;
    size_t nw = fwrite(buf, size, nitems, fp);

    if (nw != nitems) {
      printf("MPI_File_write_all(): incorrect number of items in fwrite()\n");
    }

    if (ferror(fp)) {
      perror("perror: ");
      printf("MPI_File_write_all() file operation failed\n");
      exit(0);
    }
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_File_write_all_begin
 *
 *****************************************************************************/

int MPI_File_write_all_begin(MPI_File fh, const void * buf, int count,
			     MPI_Datatype datatype) {

  /* We are going to do it here and throw away the status */

  MPI_Status status = {0};

  MPI_File_write_all(fh, buf, count, datatype, &status);

  return 0;
}

/*****************************************************************************
 *
 *  MPI_File_write_all_end
 *
 *****************************************************************************/

int MPI_File_write_all_end(MPI_File fh, const void * buf, MPI_Status * status) {

  /* A real implementation returns the number of bytes written in the
   * status object. */

  return 0;
}

#endif /* _DO_NOT_INCLUDE_MPI2_INTERFACE */

/*****************************************************************************
 *
 *  mpi_is_valid_comm
 *
 *****************************************************************************/

int mpi_is_valid_comm(MPI_Comm comm) {

  if (comm < MPI_COMM_WORLD || comm >= MAX_CART_COMM) return 0;

  return 1;
}

/*****************************************************************************
 *
 *  mpi_data_type_add
 *
 *  Add a record of active data type. At the moment we just have the
 *  extent in bytes to allow copy of contiguous types.
 *
 *****************************************************************************/

static int mpi_data_type_add(mpi_info_t * ctxt, const data_t * dt,
			     MPI_Datatype * newtype) {

  assert(ctxt);
  assert(dt);
  assert(newtype);

  ctxt->ndatatype += 1;

  if (ctxt->ndatatype >= MAX_USER_DT) {
    /* Run out of handles */
    printf("INTERNAL ERROR: run out of handles\n");
  }

  {
    int handle = ctxt->dtfreelist[ctxt->ndatatype];

    if (handle > ctxt->ndatatypelast) ctxt->ndatatypelast = handle;

    ctxt->dt[handle] = *dt;
    ctxt->dt[handle].handle = handle;
    *newtype = handle;
  }

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  mpi_data_type_free
 *
 *****************************************************************************/

static int mpi_data_type_free(mpi_info_t * ctxt, MPI_Datatype * handle) {

  assert(ctxt);
  assert(handle);

  int index = *handle;

  assert(index != 0); /* Not MPI_DATATYPE_NULL */
  assert(ctxt->dt[index].commit);

  ctxt->dt[index].handle = MPI_DATATYPE_NULL;
  ctxt->dt[index].commit = 0;

  /* Update free list */
  ctxt->dtfreelist[ctxt->ndatatype] = index;
  ctxt->ndatatype -= 1;

  *handle = MPI_DATATYPE_NULL;

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  mpi_data_type_handle
 *
 *  Valid handle to valid index.
 *
 *****************************************************************************/

static int mpi_data_type_handle(mpi_info_t * ctxt, MPI_Datatype handle) {

  int index = MPI_DATATYPE_NULL;

  assert(ctxt);
  assert(handle >= 0);

  if (handle <= ctxt->ndatatypelast) {
    index = ctxt->dt[handle].handle;
  }

  return index;
}

/*****************************************************************************
 *
 *  mpi_file_handle_retain
 *
 *  Implementation of file open.
 *  Success returns a valid MPI_FIle handle.
 *
 *****************************************************************************/

static MPI_File mpi_file_handle_retain(mpi_info_t * mpi, FILE * fp) {
  
  MPI_File fh = MPI_FILE_NULL;

  assert(mpi);
  assert(fp);

  for (int ih = 1; ih < MAX_USER_FILE; ih++) {
    if (mpi->filelist[ih].fp == NULL) {
      fh = ih;
      break;
    }
  }

  if (fh == MPI_FILE_NULL) {
    printf("Run out of MPI file handles\n");
    exit(0);
  }

  /* Record the pointer against the handle */
  mpi->filelist[fh].fp = fp;

  return fh;
}

/*****************************************************************************
 *
 *  mpi_file_handle_release
 *
 *  Release handle, and return the file pointer.
 *
 *****************************************************************************/

static FILE *  mpi_file_handle_release(mpi_info_t * mpi, MPI_File fh) {

  FILE * fp = NULL;

  assert(mpi);

  if (1 <= fh && fh < MAX_USER_FILE) {
    fp = mpi->filelist[fh].fp;
    mpi->filelist[fh].fp = NULL; /* Release */
  }

  return fp;
}

/*****************************************************************************
 *
 *  mpi_file_handle_to_fp
 *
 *  Valid handles return relevant FILE * fp (or NULL).
 *
 *****************************************************************************/

static FILE * mpi_file_handle_to_fp(mpi_info_t * mpi, MPI_File fh) {

  FILE * fp = NULL;

  assert(mpi);

  if (1 <= fh && fh < MAX_USER_FILE) {
    fp = mpi->filelist[fh].fp;
  }

  return fp;
}
