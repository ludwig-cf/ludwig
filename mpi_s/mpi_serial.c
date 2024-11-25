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
#include <errno.h>
#include <stdarg.h>
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
  int          commit;       /* Committed? */
  int          flavour;      /* Contiguous types only at present */
  MPI_Aint     lb;           /* Lower bound argument */
  int          stride;       /* Stride argument */
};

struct internal_file_view_s {
  MPI_Comm comm;                /* File communicator */
  FILE * fp;                    /* file pointer */
  MPI_Offset   disp;            /* e.g., from MPI_File_set_view() */
  MPI_Datatype etype;
  MPI_Datatype filetype;
  char datarep[MPI_MAX_DATAREP_STRING];
  char errorstr[MPI_MAX_ERROR_STRING];
};

typedef struct mpi_info_s mpi_info_t;

struct mpi_info_s {
  int initialised;               /* MPI initialised */
  int ncart;                     /* Number of Cartesian communicators */
  int period[MAX_CART_COMM][3];  /* Periodic Cartesisan per communicator */
  int reorder[MAX_CART_COMM];    /* Reorder arguments per Cartesian comm */
  int ndatatype;                 /* Current number of data types */
  data_t dt[MAX_USER_DT];        /* Internal information per data type */
  int ndatatypelast;             /* Current free list extent */
  int dtfreelist[MAX_USER_DT];   /* Free list */

  file_t filelist[MAX_USER_FILE]; /* MPI_File information for open files */

  /* The following arguments are recorded but not used */
  int key;
  int commute;
};

static mpi_info_t * mpi_info_ = NULL;

static void mpi_copy(void * send, void * recv, int count, MPI_Datatype type);
static int mpi_sizeof(MPI_Datatype type);
static int mpi_sizeof_user(MPI_Datatype handle);
static int mpi_data_type_add(mpi_info_t * ctxt, const data_t * dt,
			     MPI_Datatype * newtype);
static int mpi_data_type_free(mpi_info_t * ctxt, MPI_Datatype * handle);
static int mpi_data_type_handle(mpi_info_t * ctxt, MPI_Datatype handle);
static int mpi_datatype_intrinsic(MPI_Datatype dt);
static int mpi_datatype_user(MPI_Datatype dt);

static MPI_File mpi_file_handle_retain(mpi_info_t * ctxt, FILE * fp);
static FILE *   mpi_file_handle_release(mpi_info_t * ctxt, MPI_File handle);
static FILE *   mpi_file_handle_to_fp(mpi_info_t * info, MPI_File handle);

/* Detect various errors */

static int mpi_err_amode(int amode);
static int mpi_err_arg(int arg);
static int mpi_err_buffer(const void * buf);
static int mpi_err_comm(MPI_Comm comm);
static int mpi_err_count(int count);
static int mpi_err_datatype(MPI_Datatype dt);
static int mpi_err_errhandler(MPI_Errhandler errhandler);
static int mpi_err_file(MPI_File file);
static int mpi_err_info(MPI_Info info);
static int mpi_err_op(MPI_Op op);
static int mpi_err_rank(int rank);
static int mpi_err_root(int root);
static int mpi_err_tag(int tag);

/* In principle, the errhandler is registered against a comm, file, etc */
/* At the moment, we have only two ... */

/* typedef MPI_Comm_errhandler_function(MPI_Comm * comm, int * ierr, ...) */
/* typdeef MPI_File_errhandler_function(MPI_File * fh,   int * ierr, ...) */

static void mpi_comm_errors_are_fatal(MPI_Comm * comm, int * ifail, ...);
static void mpi_file_errors_return(MPI_File * file, int * ifail, ...);

#define ERR_IF_MPI_NOT_INITIALISED(fn)					\
  {									\
    if (mpi_info_ == NULL) {						\
      /* Always illegal; abort */					\
      printf("The %s function was called before either MPI_Init() or"	\
	     "MPI_Init_thread(). This is illegal.", fn);		\
      exit(-1);								\
    }									\
  }

/* Macros for argument checking expected to be in a routine of the form: */
/*
 * {
 *   int ifail = MPI_SUCCESS;
 *   MPI_Com self = MPI_COMM_SELF;
 *
 *   MACRO(self, ...);
 *
 *   err:
 *   return ifail;
 * }
 *   The comm and file arguments of the macro must be lvalues, as the address
 *   is taken to call the error handler.
 */

#define ERR_IF_COMM_MPI_ERR_COMM(comm, fn)				\
  {									\
    ifail = mpi_err_comm(comm);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid comm", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fn)			\
  {									\
    ifail = mpi_err_buffer(buf);					\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid buffer", fn); \
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fn)			\
  {									\
    ifail = mpi_err_count(count);					\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s, invalid count", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_DATATYPE(comm, dt, fn)			\
  {									\
    ifail = mpi_err_datatype(dt);					\
    mpi_comm_errors_are_fatal(&comm, &ifail,				\
			      "%s: invalid datatype (%d)", fn, dt);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_INFO(comm, info, fn)	 \
  {							 \
    ifail = mpi_err_info(info);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid info", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_OP(comm, op, fn)				\
  {									\
    ifail = mpi_err_op(op);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid op", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_RANK(comm, rank, fn)			\
  {									\
    ifail = mpi_err_rank(rank);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid rank", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_ROOT(comm, root, fn)			\
  {									\
    ifail = mpi_err_root(root);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid root", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fn)				\
  {									\
    ifail = mpi_err_tag(tag);						\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid tag", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_ARG(comm, arg, fn)				\
  {									\
    ifail = mpi_err_arg((arg));					\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: %s", fn, #arg);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_COMM_MPI_ERR_ERRHANDLER(comm, errhandler, fn)		\
  {									\
    ifail = mpi_err_errhandler(errhandler);				\
    mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid errhandler", fn); \
    if (ifail != MPI_SUCCESS) goto err;					\
  }

/* MPI_File routine argument error checkers */

#define ERR_IF_FILE_MPI_ERR_COMM(file, comm, fn)			\
  {									\
    ifail = mpi_err_comm(comm);						\
    mpi_file_errors_return(&file, &ifail, "%s: invalid comm", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_ARG(file, arg, fn)				\
  {									\
    ifail = mpi_err_arg((arg));						\
    mpi_file_errors_return(&file, &ifail, "%s: %s", fn, #arg);		\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_AMODE(file, amode, func)			\
  {									\
    ifail = mpi_err_amode(amode);					\
    mpi_file_errors_return(&file, &ifail, "%s: invalid amode", func);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_INFO(file, info, func) 			\
  {									\
    ifail = mpi_err_info(info);						\
    mpi_file_errors_return(&file, &ifail, "%s: invalid info", func);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_FILE(file, func)				\
  {									\
    ifail = mpi_err_file(file);						\
    mpi_file_errors_return(&file, &ifail, "%s: invalid file", func);	\
    if (ifail != MPI_SUCCESS)  goto err;				\
  }

#define ERR_IF_FILE_MPI_ERR_BUFFER(file, buf, func)			\
  {									\
    ifail = mpi_err_buffer(buf);					\
    mpi_file_errors_return(&file, &ifail, "%s: invalid buffer", func);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_COUNT(file, count, fn)			\
  {									\
    ifail = mpi_err_count(count);					\
    mpi_file_errors_return(&file, &ifail, "%s: invalid count", fn);	\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

#define ERR_IF_FILE_MPI_ERR_DATATYPE(file, datatype, func)		\
  {									\
    ifail = mpi_err_datatype(datatype);					\
    mpi_file_errors_return(&file, &ifail, "%s: invalid datatype", func);\
    if (ifail != MPI_SUCCESS) goto err;					\
  }

/*****************************************************************************
 *
 *  MPI_Error_string
 *
 *****************************************************************************/

int MPI_Error_string(int errorcode, char * string, int * resultlen) {

  int ifail = MPI_SUCCESS;
  const char * msg = NULL;

  /* May be called before MPI_Init() */
  /* "errcode" is code or class ...  */

  switch (errorcode) {
  case MPI_SUCCESS:
    msg = "MPI_SUCCESS: success";
    break;
  case MPI_ERR_ACCESS:
    msg = "MPI_ERR_ACCESS: permission denied";
    break;
  case MPI_ERR_AMODE:
    msg = "MPI_ERR_AMODE: invalid mode argument";
    break;
  case MPI_ERR_ARG:
    msg = "MPI_ERR_AMODE: invalid argument of naother kind";
    break;
  case MPI_ERR_BUFFER:
    msg = "MPI_ERR_BUFFER: invalid buffer pointer argument";
    break;
  case MPI_ERR_COMM:
    msg = "MPI_ERR_COMM: invalid communicator argument";
    break;
  case MPI_ERR_COUNT:
    msg = "MPI_ERR_COUNT: invalid count argument";
    break;
  case MPI_ERR_DATATYPE:
    msg = "MPI_ERR_DATATYPE: invalid datatype";
    break;
  case MPI_ERR_INFO:
    msg = "MPI_ERR_INFO: invalid info argument";
    break;
  case MPI_ERR_ERRHANDLER:
    msg = "MPI_ERR_ERRHANDLER: invalid error handler";
    break;
  case MPI_ERR_INTERN:
    msg = "MPI_ERR_INTERN: internal (implementation) error";
    break;
  case MPI_ERR_IO:
    msg = "MPI_ERR_IO: other i/o error";
    break;
  case MPI_ERR_FILE:
    msg = "MPI_ERR_FILE: invalid file handle";
    break;
  case MPI_ERR_NO_SUCH_FILE:
    msg = "MPI_ERR_NO_SUCH_FILE: file does not exist";
    break;
  case MPI_ERR_OP:
    msg = "MPI_ERR_OP: invalid operation";
    break;
  case MPI_ERR_RANK:
    msg = "MPI_ERR_RANK: invalid rank";
    break;
  case MPI_ERR_ROOT:
    msg = "MPI_ERR_ROOT: invalid root argument";
    break;
  case MPI_ERR_TAG:
    msg = "MPI_ERR_TAG: invalid tag";
    break;
  case MPI_ERR_LASTCODE:
    msg = "MPI_ERR_LASTCODE: last error message code";
    break;
  default:
    /* We say an unrecognised code is and unknown error ... */
    msg = "MPI_ERR_UNKNOWN: unknown error";
  }

  if (string) {
    strncpy(string, msg, MPI_MAX_ERROR_STRING);
    if (resultlen) *resultlen = strlen(msg);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Barrier
 *
 *****************************************************************************/

int MPI_Barrier(MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Barrier";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Bcast()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buffer, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_ROOT(comm, root, fname);

  /* no operation required */

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Init
 *
 *****************************************************************************/

int MPI_Init(int * argc, char *** argv) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Init()";

  ERR_IF_COMM_MPI_ERR_ARG(self, argc == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, argv == NULL, fname);

  mpi_info_ = (mpi_info_t *) calloc(1, sizeof(mpi_info_t));
  assert(mpi_info_);

  mpi_info_->initialised = 1;

  /* User data type handles: reserve dt[0].handle = 0 for MPI_DATATYPE_NULL */
  /* Otherwise, user data type handles are indexed 1, 2, 3, ... */
  /* Initialise the free list. */

  assert(MPI_DATATYPE_NULL == 0);

  for (int n = 0; n < MAX_USER_DT; n++) {
    data_t null = {.handle = 0, .bytes = 0, .commit = 0, .flavour = 0};
    mpi_info_->dt[n] = null;
    mpi_info_->dtfreelist[n] = n;
  }
  mpi_info_->ndatatype = 0;
  mpi_info_->ndatatypelast = 0;

  for (int ih = 0; ih < MAX_USER_FILE; ih++) {
    mpi_info_->filelist[ih].fp    = NULL;
    mpi_info_->filelist[ih].disp  = 0;
    mpi_info_->filelist[ih].etype = MPI_BYTE;
    mpi_info_->filelist[ih].filetype = MPI_BYTE;
    strncpy(mpi_info_->filelist[ih].datarep, "native", MPI_MAX_DATAREP_STRING);
    strncpy(mpi_info_->filelist[ih].errorstr, "", MPI_MAX_ERROR_STRING);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Init_thread
 *
 *****************************************************************************/

int MPI_Init_thread(int * argc, char *** argv, int required, int * provided) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Init_thread()";

  ERR_IF_COMM_MPI_ERR_ARG(self, argc == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, argv == NULL, fname);
  /* required; see below */
  ERR_IF_COMM_MPI_ERR_ARG(self, provided == NULL, fname);

  MPI_Init(argc, argv);

  /* MPI_THREAD_MULTIPLE is not available */

  if (MPI_THREAD_SINGLE <= required && required <= MPI_THREAD_MULTIPLE) {
    *provided = required;
    if (required == MPI_THREAD_MULTIPLE) *provided = MPI_THREAD_SERIALIZED;
  }
  else {
    ifail = MPI_ERR_ARG;
    mpi_comm_errors_are_fatal(&self, &ifail,
			      "%s: required level unrecognised", fname);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Initialized
 *
 *****************************************************************************/

int MPI_Initialized(int * flag) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Initialised()";

  ERR_IF_COMM_MPI_ERR_ARG(self, flag == NULL, fname);

  *flag = (mpi_info_ != NULL); /* A sufficient condition */

 err:
  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  MPI_Finalize
 *
 *****************************************************************************/

int MPI_Finalize(void) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Finalize";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  assert(mpi_info_->ndatatype == 0); /* All released */

  free(mpi_info_);
  mpi_info_ = NULL;

  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_group
 *
 *****************************************************************************/

int MPI_Comm_group(MPI_Comm comm, MPI_Group * group) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_group";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, group == NULL, fname);

  *group = 0;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_rank
 *
 *****************************************************************************/

int MPI_Comm_rank(MPI_Comm comm, int * rank) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_rank";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, rank == NULL, fname);

  *rank = 0;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_size
 *
 *****************************************************************************/

int MPI_Comm_size(MPI_Comm comm, int * size) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_size";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, size == NULL, fname);

  *size = 1;

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  MPI_Comm comm = MPI_COMM_SELF;
  const char * fname = "MPI_Comm_compare()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, result == NULL, fname)

  *result = MPI_UNEQUAL;
  if (!mpi_err_comm(comm1) && !mpi_err_comm(comm2)) {
    *result = MPI_CONGRUENT;
    if (comm1 == comm2) *result = MPI_IDENT;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Abort
 *
 *****************************************************************************/

int MPI_Abort(MPI_Comm comm, int code) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Abort()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);

 err:
  exit(code);

  return ifail;
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

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Send";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, dest, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);

  printf("MPI_Send should not be called in serial.\n");
  exit(0);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Recv
 *
 *****************************************************************************/

int MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source,
	     int tag, MPI_Comm comm, MPI_Status * status) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Recv()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, source, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);

  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = MPI_ERR_INTERN;
  mpi_comm_errors_are_fatal(&comm, &ifail, "%s: cannot call in serial", fname);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Irecv
 *
 *****************************************************************************/

int MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source,
	     int tag, MPI_Comm comm, MPI_Request * request) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Irecv()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, source, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, request == NULL, fname);

  *request = tag;

 err:
  return ifail;
}


/*****************************************************************************
 *
 *  MPI_Ssend
 *
 *****************************************************************************/

int MPI_Ssend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Ssend()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, dest, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);

  printf("MPI_Ssend should not be called in serial\n");
  exit(0);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Isend
 *
 *****************************************************************************/

int MPI_Isend(void * buf, int count, MPI_Datatype datatype, int dest,
	      int tag, MPI_Comm comm, MPI_Request * request) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Isend()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, dest, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, request == NULL, fname);

  *request = tag;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Issend
 *
 *****************************************************************************/

int MPI_Issend(void * buf, int count, MPI_Datatype datatype, int dest,
	       int tag, MPI_Comm comm, MPI_Request * request) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Issend()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, buf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, datatype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, dest, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, request == NULL, fname);

  printf("MPI_Issend should not be called in serial\n");
  exit(0);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Waitall
 *
 *****************************************************************************/

int MPI_Waitall(int count, MPI_Request * requests, MPI_Status * statuses) {

  int ifail = MPI_SUCCESS;
  MPI_Comm comm = MPI_COMM_SELF;
  const char * fname = "MPI_Waitall()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, requests == NULL, fname);

  if (statuses != MPI_STATUSES_IGNORE) {
    statuses[0].MPI_ERROR = MPI_SUCCESS;
  }

 err:
  return ifail;
}


/*****************************************************************************
 *
 *  MPI_Waitany
 *
 *****************************************************************************/

int MPI_Waitany(int count, MPI_Request requests[], int * index,
		MPI_Status * status) {

  int ifail = MPI_SUCCESS;
  MPI_Comm comm = MPI_COMM_SELF;
  const char * fname = "MPI_Waitany()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, index == NULL, fname);

  *index = MPI_UNDEFINED;

  for (int ireq = 0; ireq < count; ireq++) {
    if (requests[ireq] != MPI_REQUEST_NULL) {
      *index = ireq;
      requests[ireq] = MPI_REQUEST_NULL;
      if (status != MPI_STATUS_IGNORE) {
	status->MPI_SOURCE = 0;
	status->MPI_TAG = requests[ireq];
      }
      break;
    }
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Probe
 *
 *****************************************************************************/

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Probe()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, source, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, tag, fname);

  ifail = MPI_ERR_INTERN;
  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = MPI_ERR_INTERN;
  mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid serial call", fname);

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_SendRecv()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, sendcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, sendtype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, dest, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, sendtag, fname);

  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, recvcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, recvtype, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, source, fname);
  ERR_IF_COMM_MPI_ERR_TAG(comm, recvtag, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, recvcount != sendcount, fname);

  ifail = MPI_ERR_INTERN;
  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = MPI_ERR_INTERN;
  mpi_comm_errors_are_fatal(&comm, &ifail, "%s: invalid serial call", fname);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Reduce
 *
 *****************************************************************************/

int MPI_Reduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
	       MPI_Op op, int root, MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Reduce()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, type, fname);
  ERR_IF_COMM_MPI_ERR_OP(comm, op, fname);
  ERR_IF_COMM_MPI_ERR_ROOT(comm, root, fname);

  /* Whatever the operation is, the result is the same ... ! */
  mpi_copy(sendbuf, recvbuf, count, type);

 err:
  return ifail;
}

/****************************************************************************
 *
 *  MPI_Allgather
 *
 ****************************************************************************/

int MPI_Allgather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
		  void * recvbuf, int recvcount, MPI_Datatype recvtype,
		  MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Allgather()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, sendcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, sendtype, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, recvcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, recvtype, fname);

  ERR_IF_COMM_MPI_ERR_ARG(comm, sendcount != recvcount, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, sendtype != recvtype, fname);

  mpi_copy(sendbuf, recvbuf, sendcount, sendtype);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Gather
 *
 ****************************************************************************/

int MPI_Gather(void * sendbuf, int sendcount, MPI_Datatype sendtype,
	       void * recvbuf, int recvcount, MPI_Datatype recvtype,
	       int root, MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Gather()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, sendcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, sendtype, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, recvcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, recvtype, fname);
  ERR_IF_COMM_MPI_ERR_ROOT(comm, root, fname);

  ERR_IF_COMM_MPI_ERR_ARG(comm, sendcount != recvcount, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, sendtype != recvtype, fname);

  mpi_copy(sendbuf, recvbuf, sendcount, sendtype);

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Gatherv()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, (void *) sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, sendcount, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, sendtype, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, displ == NULL, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, recvtype, fname);
  ERR_IF_COMM_MPI_ERR_ROOT(comm, root, fname);

  ERR_IF_COMM_MPI_ERR_ARG(comm, sendtype != recvtype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, sendcount != recvcounts[0], fname);

  mpi_copy((void *) sendbuf, recvbuf, sendcount, sendtype);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Allreduce
 *
 *****************************************************************************/

int MPI_Allreduce(void * sendbuf, void * recvbuf, int count, MPI_Datatype type,
		  MPI_Op op, MPI_Comm comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Allreduce()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, sendbuf, fname);
  ERR_IF_COMM_MPI_ERR_BUFFER(comm, recvbuf, fname);
  ERR_IF_COMM_MPI_ERR_COUNT(comm, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(comm, type, fname);
  ERR_IF_COMM_MPI_ERR_OP(comm, op, fname);

  if (sendbuf != MPI_IN_PLACE) {
    mpi_copy(sendbuf, recvbuf, count, type);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_split
 *
 *  Return the original communicator as the new communicator.
 *
 *  The colur argument can be MPI_UNDEFINED, which may be negative.
 *  With MPI_UNDEFINED -999, this will cause an error at the moment.
 *
 *****************************************************************************/

int MPI_Comm_split(MPI_Comm comm, int colour, int key, MPI_Comm * newcomm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_split()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, colour < 0, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, newcomm == NULL, fname);

  /* Allow that a split Cartesian communicator is different */
  /* See MPI_Comm_compare() */

  mpi_info_->key = key;
  *newcomm = MPI_COMM_WORLD;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_split_type
 *
 *****************************************************************************/

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info,
			MPI_Comm * newcomm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_split_type()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, split_type != MPI_COMM_TYPE_SHARED, fname);
  /* key controls rank assignment; no constraints */
  ERR_IF_COMM_MPI_ERR_INFO(comm, info, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, newcomm == NULL, fname);

  mpi_info_->key = key;
  *newcomm = comm;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_free
 *
 *****************************************************************************/

int MPI_Comm_free(MPI_Comm * comm) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Comm_free()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, comm == NULL, fname);
  ERR_IF_COMM_MPI_ERR_COMM(*comm, fname);

  /* Mark Cartesian communicators as free */

  if (*comm > MPI_COMM_SELF) {
    mpi_info_->ncart -= 1;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Comm_dup
 *
 *  Just return the old one.
 *
 *****************************************************************************/

int MPI_Comm_dup(MPI_Comm oldcomm, MPI_Comm * newcomm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_dup()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(oldcomm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(oldcomm, newcomm == NULL, fname);

  *newcomm = oldcomm;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_indexed
 *
 *****************************************************************************/

int MPI_Type_indexed(int count, int * array_of_blocklengths,
		     int * array_of_displacements, MPI_Datatype oldtype,
		     MPI_Datatype * newtype) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_indexed()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(self, count, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_blocklengths == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_displacements == NULL, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, oldtype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = 0;
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Can't do displacements at moment */

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_contiguous
 *
 *****************************************************************************/

int MPI_Type_contiguous(int count, MPI_Datatype old, MPI_Datatype * newtype) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_contiguous()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(self, count, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, old, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = mpi_sizeof(old)*count;  /* contiguous */
    dt.commit  = 0;
    dt.flavour = DT_CONTIGUOUS;

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_commit
 *
 *****************************************************************************/

int MPI_Type_commit(MPI_Datatype * type) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_commit()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, type == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, *type == MPI_DATATYPE_NULL, fname);

  {
    int handle = *type;

    if (handle < 0) {
      ifail = MPI_ERR_ARG;
      mpi_comm_errors_are_fatal(&self, &ifail, "%s: intrinsic type!", fname);
    }

    if (handle > mpi_info_->ndatatypelast) {
      ifail = MPI_ERR_DATATYPE;
      mpi_comm_errors_are_fatal(&self, &ifail, "unrecognised datatype", fname);
    }

    assert(mpi_info_->dt[handle].handle == handle);
    mpi_info_->dt[handle].commit = 1;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_free
 *
 *****************************************************************************/

int MPI_Type_free(MPI_Datatype * type) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_free()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, type == NULL, fname);

  mpi_data_type_free(mpi_info_, type);
  assert(*type == MPI_DATATYPE_NULL);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_vector
 *
 *****************************************************************************/

int MPI_Type_vector(int count, int blocklength, int stride,
		    MPI_Datatype oldtype, MPI_Datatype * newtype) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_vector()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(self, count, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, blocklength < 0, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, oldtype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = count*blocklength*mpi_sizeof(oldtype);
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Can't do strided copy */
    dt.stride  = stride;

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Cart_create
 *
 *****************************************************************************/

int MPI_Cart_create(MPI_Comm oldcomm, int ndims, int * dims, int * periods,
		    int reorder, MPI_Comm * newcomm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_create";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(oldcomm, fname);
  /* standard doesn't put any constraint on ndims */
  ERR_IF_COMM_MPI_ERR_ARG(oldcomm, dims == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(oldcomm, periods == NULL, fname);
  /* reorder is logical; no constraints */
  ERR_IF_COMM_MPI_ERR_ARG(oldcomm, newcomm == NULL, fname);

  mpi_info_->ncart += 1;

  {
    /* Only Cartesian comms have handles above MPI_COM_SELF */
    /* See also MPI_Comm_free() */
    int icart = MPI_COMM_SELF + mpi_info_->ncart;

    if (icart >= MAX_CART_COMM) {
      ifail = MPI_ERR_INTERN;
      mpi_comm_errors_are_fatal(&oldcomm, &ifail,
				"MPI_Cart_create(): out of handles");
      goto err;
    }

    /* Record periodity, reorder */

    for (int n = 0; n < ndims; n++) {
      mpi_info_->period[icart][n] = periods[n];
    }
    mpi_info_->reorder[icart] = reorder;

    *newcomm = icart;
  }

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_get()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, dims == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, periods == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, coords == NULL, fname);

  for (int n = 0; n < maxdims; n++) {
    dims[n] = 1;
    periods[n] = mpi_info_->period[comm][n];
    coords[n] = 0;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 * MPI_Cart_coords
 *
 *****************************************************************************/

int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int * coords) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_coords()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_RANK(comm, rank, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, coords == NULL, fname);

  for (int d = 0; d < maxdims; d++) {
    coords[d] = 0;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Cart_rank
 *
 *  Set the Cartesian rank to zero.
 *
 *****************************************************************************/

int MPI_Cart_rank(MPI_Comm comm, int * coords, int * rank) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_rank()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, coords == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, rank == NULL, fname);

  *rank = 0;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Cart_shift
 *
 *****************************************************************************/

int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source,
		   int * rank_dest) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_shift()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, rank_source == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, rank_dest   == NULL, fname);

  *rank_source = 0;
  *rank_dest = 0;

  /* Non periodic directions */
  if (disp != 0 && mpi_info_->period[comm][direction] != 1) {
    *rank_source = MPI_PROC_NULL;
    *rank_dest   = MPI_PROC_NULL;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Cart_sub
 *
 *****************************************************************************/

int MPI_Cart_sub(MPI_Comm comm, int * remain_dims, MPI_Comm * new_comm) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Cart_sub()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, remain_dims == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(comm, new_comm == NULL, fname);

  *new_comm = comm;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Dims_create
 *
 *****************************************************************************/

int MPI_Dims_create(int nnodes, int ndims, int * dims) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Dims_create()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, nnodes != 1, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, dims == NULL, fname);

  for (int d = 0; d < ndims; d++) {
    dims[d] = 1;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Op_create
 *
 *****************************************************************************/

int MPI_Op_create(MPI_User_function * function, int commute, MPI_Op * op) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Op_create()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, function == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, op == NULL, fname);

  /* commute is logical */

  mpi_info_->commute = commute;
  *op = MPI_SUM;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Op_free
 *
 *****************************************************************************/

int MPI_Op_free(MPI_Op * op) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Op_free()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, op == NULL, fname);

  *op = MPI_OP_NULL;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  Internal
 *
 *****************************************************************************/

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

  assert(mpi_datatype_intrinsic(type) || mpi_datatype_user(type));

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
    /* Not implementend */
    break;
  default:
    /* ... user type */
    size = mpi_sizeof_user(type);
  }

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
  int index = mpi_data_type_handle(mpi_info_, handle);

  assert(index >= MPI_COMM_NULL); /* not intrinsic */

  if (index == MPI_DATATYPE_NULL) {
    printf("mpi_sizeof_dt: NULL data type %d\n", handle);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  else {
    sz = mpi_info_->dt[index].bytes;
  }

  return sz;
}

/*****************************************************************************
 *
 *  MPI_Comm_set_errhandler
 *
 *****************************************************************************/

int MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Comm_set_errhandler()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(comm, fname);
  ERR_IF_COMM_MPI_ERR_ERRHANDLER(comm, errhandler, fname);

  /* Only errhandler == MPI_ERRORS_ARE_FATAL available in comm */

 err:
  return ifail;
}

#ifdef _DO_NOT_INCLUDE_MPI2_INTERFACE
/*
 * The following are removed from MPI3... and have an appropriate
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
 *  Supersedes MPI_Address
 *
 *****************************************************************************/

int MPI_Get_address(const void * location, MPI_Aint * address) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Get_address()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, location == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, address  == NULL, fname);

  *address = (MPI_Aint) location;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Group_translate_ranks
 *
 *****************************************************************************/

int MPI_Group_translate_ranks(MPI_Group grp1, int n, const int * ranks1,
			      MPI_Group grp2, int * ranks2) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_Group_translate_ranks()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COMM(grp1, fname);
  ERR_IF_COMM_MPI_ERR_ARG(grp1, ranks1 == NULL, fname);
  ERR_IF_COMM_MPI_ERR_COMM(grp2, fname);
  ERR_IF_COMM_MPI_ERR_ARG(grp2, ranks2 == NULL, fname);

  memcpy(ranks2, ranks1, n*sizeof(int));

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_create_resized
 *
 *****************************************************************************/

int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
			    MPI_Datatype * newtype) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_create_resized()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, oldtype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

  {
    data_t dt = {0};

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = extent;
    dt.commit  = 0;
    dt.flavour = DT_NOT_IMPLEMENTED; /* Should be  old.flavour */
    dt.lb      = lb;

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_create_struct
 *
 *  Supersedes MPI_Type_struct()
 *
 *****************************************************************************/

int MPI_Type_create_struct(int count, int array_of_blocklengths[],
			   const MPI_Aint array_of_displacements[],
			   const MPI_Datatype array_of_types[],
			   MPI_Datatype * newtype) {
  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_create_struct()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_COUNT(self, count, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_blocklengths == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_displacements == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_types == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

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

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_get_extent
 *
 *****************************************************************************/

int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint * lb,
			MPI_Aint * extent) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_get_extent()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, datatype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, lb == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, extent == NULL, fname);

  /* Special case MPI_PACKED not implemented ... */
  ERR_IF_COMM_MPI_ERR_ARG(self, datatype == MPI_PACKED, fname);

  *lb = 0;
  *extent = mpi_sizeof(datatype);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_Type_size
 *
 *****************************************************************************/

int MPI_Type_size(MPI_Datatype datatype, int * sz) {

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_size()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, datatype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, sz == NULL, fname);

  /* Special case MPI_PACKED not implemented ... */
  ERR_IF_COMM_MPI_ERR_ARG(self, datatype == MPI_PACKED, fname);

  *sz = mpi_sizeof(datatype);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_open
 *
 *****************************************************************************/

int MPI_File_open(MPI_Comm comm, const char * filename, int amode,
		  MPI_Info info, MPI_File * fh) {

  int ifail = MPI_SUCCESS;
  MPI_File file = MPI_FILE_NULL;
  const char * fname = "MPI_File_open()";

  FILE * fp = NULL;
  const char * fdmode = NULL;

  /* Default file error handler responsible ... */

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_COMM(file, comm, fname);
  ERR_IF_FILE_MPI_ERR_ARG(file, filename == NULL, fname);
  ERR_IF_FILE_MPI_ERR_AMODE(file, amode, fname);
  ERR_IF_FILE_MPI_ERR_INFO(file, info, fname);
  ERR_IF_FILE_MPI_ERR_ARG(file, fh == NULL, fname);

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
    /* Fail in fopen() => errno is set (many possible values ...) */
    if (errno ==  ENOENT) {
      ifail = MPI_ERR_NO_SUCH_FILE;
      mpi_file_errors_return(&file, &ifail,
			     "MPI_File_open(): no such file %s",
			     filename);
    }
    else {
      ifail = MPI_ERR_IO;
      mpi_file_errors_return(&file, &ifail,
			     "MPI_File_open(): failed file %s mode %s",
			     filename, fdmode);
    }
    goto err;
  }

  {
    /* Generate a new file handle */
    file = mpi_file_handle_retain(mpi_info_, fp);
    if (file == MPI_FILE_NULL) {
      /* Internal error; run out of file handles */
      ifail = MPI_ERR_INTERN;
      mpi_file_errors_return(&file, &ifail,
			     "MPI_File_open(): run out of handles");
      fclose(fp);
      goto err;
    }
    mpi_info_->filelist[file].comm = MPI_COMM_SELF;
    *fh = file;
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_close
 *
 *****************************************************************************/

int MPI_File_close(MPI_File * fh) {

  int ifail = MPI_SUCCESS;
  MPI_File file = MPI_FILE_NULL;
  const char * fname = "MPI_File_close()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_ARG(file, fh == NULL, fname);
  ERR_IF_FILE_MPI_ERR_FILE(*fh, fname);

  /* File handle is now validated ... */
  {
    FILE * fp = mpi_file_handle_release(mpi_info_, *fh);
    fclose(fp);
  }

  *fh = MPI_FILE_NULL;

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_delete
 *
 *****************************************************************************/

int MPI_File_delete(const char * filename, MPI_Info info) {

  int ifail = MPI_SUCCESS;
  MPI_File file = MPI_FILE_NULL;
  const char * fname = "MPI_File_delete()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_ARG(file, filename == NULL, fname);
  ERR_IF_FILE_MPI_ERR_INFO(file, info, fname);

  /* remove() returns 0 on success, -1 otherwise. errno is set. */

  if (remove(filename) != 0) {

    /* The standard gives the following options ...
     * MPI_ERR_NO_SUCH_FILE if the file does not exist; or
     * MPI_ERR_FILE_IN_USE or MPI_ERR_ACCESS. We use the latter. */

    if (errno == ENOENT) {
      ifail = MPI_ERR_NO_SUCH_FILE;
      mpi_file_errors_return(&file, &ifail, "MPI_Delete(): no such file");
    }
    else {
      ifail = MPI_ERR_ACCESS;
      mpi_file_errors_return(&file, &ifail, "MPI_Delete(): access error");
    }
  }

 err:
  return ifail;
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

  int ifail = MPI_SUCCESS;
  MPI_Comm self = MPI_COMM_SELF;
  const char * fname = "MPI_Type_create_subarray";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, ndims <= 0, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_subsizes == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, array_of_starts == NULL, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, order != MPI_ORDER_C, fname);
  ERR_IF_COMM_MPI_ERR_DATATYPE(self, oldtype, fname);
  ERR_IF_COMM_MPI_ERR_ARG(self, newtype == NULL, fname);

  /* Should really be ... */
  assert(order == MPI_ORDER_C || order == MPI_ORDER_FORTRAN);

  /* Assume this is a contiguous block of elements of oldtype */

  {
    int nelements = 1;
    data_t dt = {0};

    for (int idim = 0; idim < ndims; idim++) {
      nelements *= array_of_sizes[idim];
    }

    dt.handle  = MPI_DATATYPE_NULL;
    dt.bytes   = mpi_sizeof(oldtype)*nelements;
    dt.commit  = 0;
    dt.flavour = DT_SUBARRAY;

    mpi_data_type_add(mpi_info_, &dt, newtype);
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_get_view
 *
 *****************************************************************************/

int MPI_File_get_view(MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype,
		      MPI_Datatype * filetype, char * datarep) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_get_view";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, disp == NULL, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, etype == NULL, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, filetype == NULL, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, datarep == NULL, fname);

  {
    file_t * file = &mpi_info_->filelist[fh];

    *disp = file->disp;
    *etype = file->etype;
    *filetype = file->filetype;
    strncpy(datarep, file->datarep, MPI_MAX_DATAREP_STRING-1);
  }

 err:
  return ifail;
}
/*****************************************************************************
 *
 *  MPI_File_set_view
 *
 *****************************************************************************/

int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
		      MPI_Datatype filetype, const char * datarep,
		      MPI_Info info) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_set_view()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_DATATYPE(fh, etype, fname);
  ERR_IF_FILE_MPI_ERR_DATATYPE(fh, filetype, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, datarep == NULL, fname);
  ERR_IF_FILE_MPI_ERR_ARG(fh, strcmp(datarep, "native") != 0, fname);
  ERR_IF_FILE_MPI_ERR_INFO(fh, info, fname);

  /* There is actually MPI_ERR_UNSUPPORTED_DATAREP */

  {
    file_t * file = &mpi_info_->filelist[fh];
    file->disp = disp;
    file->etype = etype;
    file->filetype = filetype;
    strncpy(file->datarep, datarep, MPI_MAX_DATAREP_STRING-1);
    /* info is currently discarded */
  }

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_read_all
 *
 *****************************************************************************/

int MPI_File_read_all(MPI_File fh, void * buf, int count,
		      MPI_Datatype datatype, MPI_Status * status) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_read_all()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_BUFFER(fh, buf, fname);
  ERR_IF_FILE_MPI_ERR_COUNT(fh, count, fname);
  ERR_IF_FILE_MPI_ERR_DATATYPE(fh, datatype, fname);

  {
    FILE * fp = mpi_file_handle_to_fp(mpi_info_, fh);

    /* Translate to a simple fread() */
    /* A short count of items indicates an error (eof or error) ... */

    size_t size   = mpi_sizeof(datatype);
    size_t nitems = count;
    size_t nread  = fread(buf, size, nitems, fp);


    if (nread < nitems) {
      ifail = MPI_ERR_IO;
      printf("MPI_File_read_all(): "); if (ferror(fp)) perror(NULL);
    }
  }

 err:
  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = ifail;

  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_write_all
 *
 *****************************************************************************/

int MPI_File_write_all(MPI_File fh, const void * buf, int count,
		       MPI_Datatype datatype, MPI_Status * status) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_write_all()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_BUFFER(fh, buf, fname);
  ERR_IF_FILE_MPI_ERR_COUNT(fh, count, fname);
  ERR_IF_FILE_MPI_ERR_DATATYPE(fh, datatype, fname);

  {
    FILE * fp = mpi_file_handle_to_fp(mpi_info_, fh);

    /* Translate to a simple fwrite() */
    /* A short count of items indicates an error ... */

    size_t size   = mpi_sizeof(datatype);
    size_t nitems = count;
    size_t nwrite = fwrite(buf, size, nitems, fp);

    if (nwrite < nitems) {
      ifail = MPI_ERR_IO;
      printf("MPI_File_write_all(): "); if (ferror(fp)) perror(NULL);
    }
  }

 err:
  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = ifail;

  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_write_all_begin
 *
 *****************************************************************************/

int MPI_File_write_all_begin(MPI_File fh, const void * buf, int count,
			     MPI_Datatype datatype) {

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_write_all_begin()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_BUFFER(fh, buf, fname);
  ERR_IF_FILE_MPI_ERR_COUNT(fh, count, fname);
  ERR_IF_FILE_MPI_ERR_DATATYPE(fh, datatype, fname);

  /* We are going to do it here and throw away the status */

  ifail = MPI_File_write_all(fh, buf, count, datatype, MPI_STATUS_IGNORE);

 err:
  return ifail;
}

/*****************************************************************************
 *
 *  MPI_File_write_all_end
 *
 *****************************************************************************/

int MPI_File_write_all_end(MPI_File fh, const void * buf, MPI_Status * status) {

  /* A real implementation returns the number of bytes written in the
   * status object. */

  int ifail = MPI_SUCCESS;
  const char * fname = "MPI_File_write_all_end()";

  ERR_IF_MPI_NOT_INITIALISED(fname);
  ERR_IF_FILE_MPI_ERR_FILE(fh, fname);
  ERR_IF_FILE_MPI_ERR_BUFFER(fh, buf, fname);

  if (status != MPI_STATUS_IGNORE) status->MPI_ERROR = MPI_SUCCESS;

 err:
  return ifail;
}

#endif /* _DO_NOT_INCLUDE_MPI2_INTERFACE */

/*****************************************************************************
 *
 *  mpi_err_comm
 *
 *****************************************************************************/

static int mpi_err_comm(MPI_Comm comm) {

  int ifail = MPI_SUCCESS;

  if (comm <  MPI_COMM_WORLD) ifail = MPI_ERR_COMM;
  if (comm >= MAX_CART_COMM)  ifail = MPI_ERR_COMM;

  return ifail;
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

static int mpi_data_type_handle(mpi_info_t * mpi, MPI_Datatype handle) {

  int index = MPI_DATATYPE_NULL;

  assert(mpi);
  assert(handle >= 0); /* i.e. MPI_DATATYPE_NULL or user data type. */

  if (handle <= mpi->ndatatypelast) {
    index = mpi->dt[handle].handle;
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

  /* Find a free handle */
  for (int ih = 1; ih < MAX_USER_FILE; ih++) {
    if (mpi->filelist[ih].fp == NULL) {
      fh = ih;
      break;
    }
  }

  if (fh != MPI_FILE_NULL) {
    /* Record the pointer against the handle */
    mpi->filelist[fh].fp = fp;
  }

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

/*****************************************************************************
 *
 *  mpi_err_file
 *
 *  Returns MPI_SUCCESS if file is a valid file handle
 *  or MPI_ERR_FILE if invalid.
 *
 *****************************************************************************/

static int mpi_err_file(MPI_File file) {

  int ifail = MPI_SUCCESS;

  assert(mpi_info_);

  if (mpi_file_handle_to_fp(mpi_info_, file) == NULL) ifail = MPI_ERR_FILE;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_count
 *
 *****************************************************************************/

static int mpi_err_count(int count) {

  int ifail = MPI_SUCCESS;

  if (count < 0) ifail = MPI_ERR_COUNT;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_datayupe
 *
 *****************************************************************************/

static int mpi_err_datatype(MPI_Datatype dt) {

  int ifail = MPI_SUCCESS;

  if (mpi_datatype_intrinsic(dt) == 0 && mpi_datatype_user(dt) == 0) {
    ifail = MPI_ERR_DATATYPE;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_arg
 *
 ****************************************************************************/

static int mpi_err_arg(int arg) {

  int ifail = MPI_SUCCESS;

  if (arg) ifail = MPI_ERR_ARG; /* (sic) arg is a condition for failure */

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_amode
 *
 *  amode is the mode argument to MPI_File_open(); this returns
 *  MPI_ERR_AMODE if amode is invalid.
 *
 *****************************************************************************/

static int mpi_err_amode(int amode) {

  int ifail = MPI_SUCCESS;

  {
    int have_rdonly = (amode & MPI_MODE_RDONLY) ? 1 : 0;
    int have_wronly = (amode & MPI_MODE_WRONLY) ? 2 : 0;
    int have_rdwr   = (amode & MPI_MODE_RDWR)   ? 4 : 0;

    int have_create = (amode & MPI_MODE_CREATE);
    int have_excl   = (amode & MPI_MODE_EXCL);

    switch (have_rdonly + have_wronly + have_rdwr) {
    case (1):
      /* Read only cannot have ... */
      if (have_create) ifail = MPI_ERR_AMODE;
      if (have_excl)   ifail = MPI_ERR_AMODE;
      break;
    case (2):
      /* Write only  */
      break;
    case (4):
      /* Read write */
      break;
    default:
      /* Not recognised */
      ifail = MPI_ERR_AMODE;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_buffer
 *
 *****************************************************************************/

static int mpi_err_buffer(const void * buf) {

  int ifail = MPI_SUCCESS;

  if (!buf) ifail = MPI_ERR_BUFFER;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_errhandler
 *
 *****************************************************************************/

static int mpi_err_errhandler(MPI_Errhandler errhandler) {

  int ifail = MPI_SUCCESS;

  if (errhandler != MPI_ERRORS_ARE_FATAL) ifail = MPI_ERR_ERRHANDLER;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_info
 *
 *****************************************************************************/

static int mpi_err_info(MPI_Info info) {

  int ifail = MPI_SUCCESS;

  if (info != MPI_INFO_NULL) {
    /* At the moment, only MPI_INFO_NULL is handled */
    ifail = MPI_ERR_INTERN;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_op
 *
 *****************************************************************************/

static int mpi_err_op(MPI_Op op) {

  int ifail = MPI_SUCCESS;

  /* Any recognised op is fine (except MPI_OP_NULL).
   * Strictly, the validity of an op depends on additional information
   * such as data type, so this is very simple ... */

  if (op < 0)        ifail = MPI_ERR_OP;
  if (op > MPI_LXOR) ifail = MPI_ERR_OP;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_rank
 *
 *****************************************************************************/

static int mpi_err_rank(int rank) {

  int ifail = MPI_SUCCESS;

  if (rank != 0)              ifail = MPI_ERR_RANK;
  if (rank == MPI_ANY_SOURCE) ifail = MPI_SUCCESS;    /* ... but this ok */
  if (rank == MPI_PROC_NULL)  ifail = MPI_SUCCESS;    /* also ok. */

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_root
 *
 *****************************************************************************/

static int mpi_err_root(int root) {

  int ifail = MPI_SUCCESS;

  if (root != 0) ifail = MPI_ERR_ROOT;

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_err_tag
 *
 *****************************************************************************/

static int mpi_err_tag(int tag) {

  int ifail = MPI_SUCCESS;

  if (tag <= 0) ifail = MPI_ERR_TAG;
  if (tag == MPI_ANY_TAG) ifail = MPI_SUCCESS; /* MPI_ANY_TAG < 0 */

  return ifail;
}

/*****************************************************************************
 *
 *  mpi_comm_errors_are_fatal
 *
 *  The first optional argument must be present, and it should be a
 *  format string suitable for a printf()-like function. Remaining
 *  arguments should be consistent with the format.
 *
 *  This is fatal if ifail != MPI_SUCCESS.
 *
 *****************************************************************************/

static void mpi_comm_errors_are_fatal(MPI_Comm * comm, int * ifail, ...) {

  assert(comm);
  assert(ifail);

  if (*ifail != MPI_SUCCESS) {
    va_list ap;
    va_start(ap, ifail);
    {
      const char * fmt = va_arg(ap, const char *);
      printf("MPI_ERRORS_ARE_FATAL: ");
      vprintf(fmt, ap);
      printf(" (comm = %d)\n", *comm);
    }
    va_end(ap);
    exit(-1);
  }

  return;
}

/*****************************************************************************
 *
 *  mpi_file_errors_return
 *
 *  I'm going to store the error string, but I don't know if there's
 *  anything to be done with it...
 *
 *****************************************************************************/

static void mpi_file_errors_return(MPI_File * file, int * ifail, ...) {

  assert(file);
  assert(ifail);

  if (*ifail != MPI_SUCCESS) {
    char * errorstr = mpi_info_->filelist[*file].errorstr;
    va_list ap;
    va_start(ap, ifail);
    {
      const char * fmt = va_arg(ap, const char *);
      vsnprintf(errorstr, MPI_MAX_ERROR_STRING-1, fmt, ap);
    }
    va_end(ap);
  }

  return;
}

/*****************************************************************************
 *
 *  mpi_datatype_intrinsic
 *
 *****************************************************************************/

static int mpi_datatype_intrinsic(MPI_Datatype dt) {

  int intrinsic = 0;

  /* Is dt an intrinsic datatype? See mpi.h */

  if (MPI_CHAR >= dt && dt >= MPI_INT64_T) intrinsic = 1;

  return intrinsic;
}

/*****************************************************************************
 *
 *  mpi_datatype_user
 *
 *****************************************************************************/

static int mpi_datatype_user(MPI_Datatype dt) {

  int isuser = 0;

  assert(mpi_info_);
  assert(MPI_DATATYPE_NULL == 0);   /* mpi_info->dt[0] is null */

  /* Is dt a valid user datatype */

  if (0 < dt && dt <= MAX_USER_DT) {
    /* check the list */
    if (mpi_info_->dt[dt].handle != MPI_DATATYPE_NULL) isuser = 1;
  }

  return isuser;
}
