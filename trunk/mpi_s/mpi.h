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

/* Defined constants (see Annex A.2) */

/* Return codes */

enum return_codes {MPI_SUCCESS};

/* Assorted constants */

#define MPI_PROC_NULL     -9;
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
int MPI_Comm_rank(MPI_Comm comm, int * rank);
int MPI_Comm_size(MPI_Comm comm, int * size);

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


int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
	       MPI_Op op, int root, MPI_Comm comm);


int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype old,
		    MPI_Datatype * new);
int MPI_Type_commit(MPI_Datatype * datatype);
int MPI_Type_free(MPI_Datatype * datatype);
int MPI_Waitall(int count, MPI_Request * array_of_requests,
		MPI_Status * array_of_statuses);
int MPI_Allreduce(void * send, void * recv, int count, MPI_Datatype type,
		  MPI_Op op, MPI_Comm comm);

int MPI_Comm_split(MPI_Comm comm, int colour, int key, MPI_Comm * newcomm);
int MPI_Comm_free(MPI_Comm * comm);

/* TODO */
/* Bindings for process topologies */

int MPI_Cart_create(MPI_Comm comm_old, int ndims, int * dims, int * periods,
		    int reoerder, MPI_Comm * comm_cart);
int MPI_Dims_create(int nnodes, int ndims, int * dims);
int MPI_Cart_rank(MPI_Comm comm, int * coords, int * rank);
int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int * coords);
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source,
		   int * rank_dest);

/* Bindings for environmental inquiry */

int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);

double MPI_Wtime(void);
double MPI_Wtick(void);

int MPI_Init(int * argc, char *** argv);
int MPI_Finalize(void);
int MPI_Initialized(int * flag);
int MPI_Abort(MPI_Comm comm, int errorcode);

#endif /* _MPI_SERIAL */
