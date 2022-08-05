/*****************************************************************************
 *
 *  mpi.h
 *
 *  Serial interface to MPI routines serving as a replacement in serial.
 *
 *  Broadly, point-to-point operations are disallowed, while
 *  global operations do nothing.
 *
 *  From an idea appearing in LAMMPS.
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MPI_SERIAL
#define LUDWIG_MPI_SERIAL

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Datatypes */

typedef int MPI_Handle;
typedef MPI_Handle MPI_Comm;
typedef MPI_Handle MPI_Group;
typedef MPI_Handle MPI_Datatype;
typedef MPI_Handle MPI_Request;
typedef MPI_Handle MPI_Op;
typedef MPI_Handle MPI_Errhandler;
typedef MPI_Handle MPI_File;
typedef MPI_Handle MPI_Info;

typedef struct {
  int MPI_SOURCE;
  int MPI_TAG;
} MPI_Status;

#define MPI_STATUS_IGNORE   ((MPI_Status *) 0)
#define MPI_STATUSES_IGNORE ((MPI_Status *) 0)

/* MPI_Aint is a signed integer. Prefer intmax_t over intptr_t as
   the latter is optional in the standard. */
/* MPI_Offset is a large integer. */

typedef intmax_t MPI_Aint;
typedef intmax_t MPI_Offset;

/* Defined constants (see Annex A.2) */

/* Return codes */

enum return_codes {MPI_SUCCESS};

/* Assorted constants */

#define MPI_PROC_NULL     -9
#define MPI_ANY_SOURCE    -10
#define MPI_ANY_TAG       -11
#define MPI_BOTTOM         0x0000
#define MPI_UNDEFINED     -999

/* Error-handling specifiers */

enum error_specifiers {MPI_ERRORS_ARE_FATAL, MPI_ERRORS_RETURN};

enum elementary_datatypes {MPI_CHAR           = -11,
			   MPI_SHORT          = -12,
			   MPI_INT            = -13,
			   MPI_LONG           = -14,
			   MPI_UNSIGNED_CHAR  = -15,
			   MPI_UNSIGNED_SHORT = -16,
			   MPI_UNSIGNED       = -17,
			   MPI_UNSIGNED_LONG  = -18,
			   MPI_FLOAT          = -19,
			   MPI_DOUBLE         = -20,
			   MPI_LONG_DOUBLE    = -21,
			   MPI_BYTE           = -22,
			   MPI_PACKED         = -23};

enum collective_operations {MPI_MAX,
			    MPI_MIN,
			    MPI_SUM,
			    MPI_PROD,
			    MPI_MAXLOC,
			    MPI_MINLOC,
			    MPI_BAND,
			    MPI_BOR,
			    MPI_BXOR,
			    MPI_LAND,
			    MPI_LOR,
			    MPI_LXOR};

/* reserved communicators */

enum reserved_communicators{MPI_COMM_WORLD, MPI_COMM_SELF};

/* NULL handles */

#define MPI_GROUP_NULL      -1
#define MPI_COMM_NULL       -2
#define MPI_DATATYPE_NULL    0
#define MPI_REQUEST_NULL    -4
#define MPI_OP_NULL         -5
#define MPI_ERRHANDLER_NULL -6
#define MPI_FILE_NULL       -7
#define MPI_INFO_NULL       -8

/* Special values */

#define MPI_IN_PLACE ((void *) 1)

/* Thread support level */

#define MPI_THREAD_SINGLE      1
#define MPI_THREAD_FUNNELED    2
#define MPI_THREAD_SERIALIZED  3
#define MPI_THREAD_MULTIPLE    4

/* MPI_ORDER  */

enum mpi_order_enum {MPI_ORDER_C, MPI_ORDER_FORTRAN};

/* MPI File amodes (bitmask) */

#define MPI_MODE_RDONLY            1
#define MPI_MODE_RDWR              2
#define MPI_MODE_WRONLY            4
#define MPI_MODE_CREATE            8
#define MPI_MODE_EXCL             16
#define MPI_MODE_DELETE_ON_CLOSE  32
#define MPI_MODE_UNIQUE_OPEN      64
#define MPI_MODE_SEQUENTIAL      128
#define MPI_MODE_APPEND          256

/* Interface */

int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void * buffer, int count, MPI_Datatype datatype, int root,
	      MPI_Comm comm);
int MPI_Comm_rank(MPI_Comm comm, int * rank);
int MPI_Comm_size(MPI_Comm comm, int * size);
int MPI_Comm_group(MPI_Comm comm, MPI_Group * grp);

int MPI_Send(void * buf, int count, MPI_Datatype type, int dest, int tag,
	     MPI_Comm comm);

int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source,
	     int tag, MPI_Comm comm, MPI_Status * status);
int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request * request);

int MPI_Ssend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm);
int MPI_Isend(void * buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request * request);
int MPI_Issend(void * buf, int count, MPI_Datatype datatype, int dest,
	       int tag, MPI_Comm comm, MPI_Request * request);


int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status);


typedef void MPI_User_function(void * invec, void * inoutvec, int * len,
			       MPI_Datatype * datatype);
int MPI_Op_create(MPI_User_function * function, int commute, MPI_Op * op);
int MPI_Op_free(MPI_Op * op);

int MPI_Sendrecv(void * sendbuf, int sendcount, MPI_Datatype sendtype,
		 int dest, int sendtag, void  *recvbuf, int recvcount,
		 MPI_Datatype recvtype, int source, MPI_Datatype recvtag,
		 MPI_Comm comm, MPI_Status * status);

int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
	       MPI_Op op, int root, MPI_Comm comm);


int MPI_Type_indexed(int count, int * array_of_blocklengths,
		     int * array_of_displacements, MPI_Datatype oldtype,
		     MPI_Datatype * newtype);
int MPI_Type_contiguous(int count, MPI_Datatype oldtype,
			MPI_Datatype * newtype);
int MPI_Type_vector(int count, int blocklength, int stride,
		    MPI_Datatype oldtype, MPI_Datatype * newtype);
int MPI_Type_commit(MPI_Datatype * datatype);
int MPI_Type_free(MPI_Datatype * datatype);
int MPI_Waitall(int count, MPI_Request * array_of_requests,
		MPI_Status * array_of_statuses);
int MPI_Waitany(int count, MPI_Request array_of_req[], int * index,
		MPI_Status * status);
int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
	       void * recvbuf, int recvcount, MPI_Datatype recvtype,
	       int root, MPI_Comm comm);
int MPI_Gatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype,
		void * recvbuf, const int * recvcounts, const int * displ,
		MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Allgather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
		  void * recvbuf, int recvcount, MPI_Datatype recvtype,
		  MPI_Comm comm);
int MPI_Allreduce(void * send, void * recv, int count, MPI_Datatype type,
		  MPI_Op op, MPI_Comm comm);

int MPI_Comm_split(MPI_Comm comm, int colour, int key, MPI_Comm * newcomm);
int MPI_Comm_free(MPI_Comm * comm);
int MPI_Comm_dup(MPI_Comm oldcomm, MPI_Comm * newcomm);

/* TODO */
/* Bindings for process topologies */

int MPI_Cart_create(MPI_Comm comm_old, int ndims, int * dims, int * periods,
		    int reoerder, MPI_Comm * comm_cart);
int MPI_Dims_create(int nnodes, int ndims, int * dims);
int MPI_Cart_get(MPI_Comm comm, int maxdims, int * dims, int * periods,
		 int * coords);
int MPI_Cart_rank(MPI_Comm comm, int * coords, int * rank);
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int * coords);
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source,
		   int * rank_dest);
int MPI_Cart_sub(MPI_Comm comm, int * remain_dims, MPI_Comm * new_comm);

/* Bindings for environmental inquiry */

int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);

double MPI_Wtime(void);
double MPI_Wtick(void);

int MPI_Init(int * argc, char *** argv);
int MPI_Init_thread(int * argc, char *** argv, int required, int * provided);
int MPI_Finalize(void);
int MPI_Initialized(int * flag);
int MPI_Abort(MPI_Comm comm, int errorcode);

/* MPI 2.0 */
/* In particular, replacements for routines removed from MPI 3 */

int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler erhandler);
int MPI_Get_address(const void * location, MPI_Aint * address);
int MPI_Group_translate_ranks(MPI_Group grp1, int n, const int * ranks1,
			      MPI_Group grp2, int * ranks2);
int MPI_Type_create_struct(int count, int * arry_of_blocklens,
			   const MPI_Aint * array_of_displacements,
			   const MPI_Datatype * array_of_datatypes,
			   MPI_Datatype * newtype);
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint ub, MPI_Aint extent,
			    MPI_Datatype * newtype);
int MPI_Type_get_extent(MPI_Datatype handle, MPI_Aint * lb, MPI_Aint *extent);

/* MPI IO related */

int MPI_File_open(MPI_Comm comm, const char * filename, int amode,
		  MPI_Info info, MPI_File * fh);
int MPI_File_close(MPI_File * fh);
int MPI_File_delete(const char * filename, MPI_Info info);
int MPI_Type_create_subarray(int ndims, const int * array_of_sizes,
			     const int * array_of_subsizes,
			     const int * array_of_starts,
			     int order,
			     MPI_Datatype oldtype,
			     MPI_Datatype * newtype);
int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
		      MPI_Datatype filetype, const char * datarep,
		      MPI_Info info);
int MPI_File_read_all(MPI_File fh, void * buf, int count,
		      MPI_Datatype datatype, MPI_Status * status);
int MPI_File_write_all(MPI_File fh, const void * buf, int count,
		       MPI_Datatype datatype, MPI_Status * status);

#ifdef __cplusplus
}
#endif

#endif /* LUDWIG_MPI_SERIAL */
