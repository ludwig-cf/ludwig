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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _MPI_SERIAL
#define _MPI_SERIAL

#ifdef __cplusplus
extern "C" {
#endif

/* Datatypes */

typedef int MPI_Handle;
typedef MPI_Handle MPI_Comm;
typedef MPI_Handle MPI_Datatype;
typedef MPI_Handle MPI_Request;
typedef MPI_Handle MPI_Op;
typedef MPI_Handle MPI_Errhandler;

typedef struct {
  int MPI_SOURCE;
  int MPI_TAG;
} MPI_Status;

typedef MPI_Handle MPI_Aint;

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

enum elementary_datatypes {MPI_CHAR,
			   MPI_SHORT,
			   MPI_INT,
			   MPI_LONG,
			   MPI_UNSIGNED_CHAR,
			   MPI_UNSIGNED_SHORT,
			   MPI_UNSIGNED,
			   MPI_UNSIGNED_LONG,
			   MPI_FLOAT,
			   MPI_DOUBLE,
			   MPI_LONG_DOUBLE,
			   MPI_BYTE,
			   MPI_PACKED};

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

/* special datatypes for constructing derived datatypes */

#define MPI_UB 0 
#define MPI_LB 0

/* reserved communicators */

enum reserved_communicators{MPI_COMM_WORLD, MPI_COMM_SELF};

/* NULL handles */

#define MPI_GROUP_NULL      -1
#define MPI_COMM_NULL       -2
#define MPI_DATATYPE_NULL   -3
#define MPI_REQUEST_NULL    -4
#define MPI_OP_NULL         -5
#define MPI_ERRHANDLER_NULL -6

/* Interface */

int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void * buffer, int count, MPI_Datatype datatype, int root,
	      MPI_Comm comm);
int MPI_Comm_rank(MPI_Comm comm, int * rank);
int MPI_Comm_size(MPI_Comm comm, int * size);

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
int MPI_Type_struct(int count, int * array_of_blocklengths,
		    MPI_Aint * array_of_displacements,
		    MPI_Datatype * array_of_types, MPI_Datatype * newtype);
int MPI_Address(void * location, MPI_Aint * address);
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
int MPI_Finalize(void);
int MPI_Initialized(int * flag);
int MPI_Abort(MPI_Comm comm, int errorcode);

/* MPI 2.0 */
/* In particular, replacements for routines removed from MPI 3 */
/* MPI_Address() -> MPI_Get_Address()
 * MPI_Type_struct() -> MPI_Type_create_struct()
 * MPI_Type_lb() and MPI_Type_ub() -> MPI_Type_get_extent()
 * See MPI 3 standard */

int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler erhandler);
int MPI_Get_address(const void * location, MPI_Aint * address);
int MPI_Type_create_struct(int count, int * arry_of_blocklens,
			   const MPI_Aint * array_of_displacements,
			   const MPI_Datatype * array_of_datatypes,
			   MPI_Datatype * newtype);
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint ub, MPI_Aint extent,
			    MPI_Datatype * newtype);

#ifdef __cplusplus
}
#endif

#endif /* _MPI_SERIAL */
