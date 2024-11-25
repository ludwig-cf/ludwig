/*****************************************************************************
 *
 *  mpi_tests
 *
 *  Tests for the serial stubs
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2024 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "mpi.h"

static MPI_Comm comm_ = MPI_COMM_WORLD;
static int test_mpi_comm_rank(void);
static int test_mpi_comm_size(void);
static int test_mpi_allreduce(void);
static int test_mpi_reduce(void);
static int test_mpi_allgather(void);
static int test_mpi_type_contiguous(void);
static int test_mpi_type_create_struct(void);
static int test_mpi_type_size(void);
static int test_mpi_type_get_extent(void);
static int test_mpi_op_create(void);
static int test_mpi_file_open(void);
static int test_mpi_file_get_view(void);
static int test_mpi_file_set_view(void);
static int test_mpi_type_create_subarray(void);
static int test_mpi_file_write_all(void);

static int test_mpi_comm_split_type(void);
static int test_mpi_cart_create(void);
static int test_mpi_cart_get(void);
static int test_mpi_dims_create(void);


/* Utilities */

/*****************************************************************************
 *
 *  util_printf_bits
 *
 *  Convenience to print a set of bits as 0 or 1.
 *
 *****************************************************************************/

void util_printf_bits(size_t size, void const * data) {

  unsigned char * bytes = (unsigned char *) data;

  for (size_t ibyte = size; ibyte-- > 0; ) {
    for (int ibit = 7; ibit >= 0; ibit--) {
      unsigned char bit = (bytes[ibyte] >> ibit) & 1;
      printf("%u", bit);
    }
  }

  printf("\n");

  return;
}

/*****************************************************************************
 *
 *  util_bits_same
 *
 *  Return 1 if two bits arrays are the same.
 *
 *****************************************************************************/

int util_bits_same(size_t size, const void * data1, const void * data2) {

  int bsame = 1;
  unsigned char * b1 = (unsigned char *) data1;
  unsigned char * b2 = (unsigned char *) data2;

  assert(b1);
  assert(b2);

  for (size_t ibyte = size; ibyte-- > 0; ) {
    for (int ibit = 7; ibit >= 0; ibit--) {
      unsigned char bit1 = (b1[ibyte] >> ibit) & 1;
      unsigned char bit2 = (b2[ibyte] >> ibit) & 1;
      if (bit1 != bit2) bsame = 0;
    }
  }

  return bsame;
}

/*****************************************************************************
 *
 *  util_double_same
 *
 *  Return 1 if two doubles are bitwise the same.
 *
 *****************************************************************************/

int util_double_same(double d1, double d2) {

  double a = d1;
  double b = d2;

  return util_bits_same(sizeof(double), &a, &b);
}


/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main (int argc, char ** argv) {

  int ireturn;

  printf("Running mpi_s tests...\n");

  ireturn = MPI_Init(&argc, &argv);
  assert (ireturn == MPI_SUCCESS);

  ireturn = test_mpi_comm_rank();
  ireturn = test_mpi_comm_size();
  ireturn = test_mpi_allreduce();
  ireturn = test_mpi_reduce();
  ireturn = test_mpi_allgather();

  test_mpi_type_contiguous();
  test_mpi_type_create_struct();
  test_mpi_type_size();
  test_mpi_type_get_extent();
  test_mpi_op_create();
  test_mpi_comm_split_type();
  test_mpi_cart_create();
  test_mpi_cart_get();
  test_mpi_dims_create();

  test_mpi_file_open();
  test_mpi_file_get_view();
  test_mpi_file_set_view();
  test_mpi_type_create_subarray();
  test_mpi_file_write_all();

  ireturn = MPI_Finalize();
  assert(ireturn == MPI_SUCCESS);

  printf("Finished mpi_s tests ok.\n");

  return ireturn;
}

/*****************************************************************************
 *
 *  test_mpi_comm_rank
 *
 *****************************************************************************/

static int test_mpi_comm_rank(void) {

  int rank = MPI_PROC_NULL;
  int ireturn;

  ireturn = MPI_Comm_rank(comm_, &rank);
  assert(ireturn == MPI_SUCCESS);
  assert(rank == 0);

  return ireturn;
}

/*****************************************************************************
 *
 *  test_mpi_comm_size
 *
 *****************************************************************************/

static int test_mpi_comm_size(void) {

  int size = 0;
  int ireturn;

  ireturn = MPI_Comm_size(comm_, &size);
  assert(ireturn == MPI_SUCCESS);
  assert(size == 1);

  return ireturn;
}

/*****************************************************************************
 *
 *  test_mpi_allreduce
 *
 *****************************************************************************/

static int test_mpi_allreduce(void) {

  int ifail = 0;
  MPI_Comm comm = MPI_COMM_WORLD;

  {
    double dsend = 1.0;
    double drecv = 0.0;
    int ireturn = MPI_Allreduce(&dsend, &drecv, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (ireturn != MPI_SUCCESS) ifail = 1;
    assert(ifail == 0);
    if (util_double_same(dsend, 1.0) == 0) ifail = 1;   /* Exactly */
    assert(ifail == 0);
    if (util_double_same(drecv, dsend) == 0) ifail = 1; /* Exactly */
    assert(ifail == 0);
  }

  {
    int isend[3] = {1, 2, 3};
    int irecv[3] = {0, 0, 0};

    int ireturn = MPI_Allreduce(isend, irecv, 3, MPI_INT, MPI_SUM, comm);
    if (ireturn != MPI_SUCCESS) ifail = 0;
    assert(ifail == 0);
    if (isend[0] != 1) ifail += 1;
    if (isend[1] != 2) ifail += 1;
    if (isend[2] != 3) ifail += 1;
    assert(ifail == 0);
    if (irecv[0] != isend[0]) ifail += 1;
    if (irecv[1] != isend[1]) ifail += 1;
    if (irecv[2] != isend[2]) ifail += 1;
    assert(ifail == 0);
  }

  {
    int irecv = 1;
    int iret  = MPI_Allreduce(MPI_IN_PLACE, &irecv, 1, MPI_INT, MPI_SUM, comm);
    if (iret != MPI_SUCCESS) ifail = 1;
    assert(ifail == 0);
    if (irecv != 1) ifail = 1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_reduce
 *
 *****************************************************************************/

static int test_mpi_reduce(void) {

  int ireturn;
  int isend[3], irecv[3];
  double dsend, drecv;
  double * dvsend, * dvrecv;

  dsend = 1.0; drecv = 0.0;
  ireturn = MPI_Reduce(&dsend, &drecv, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(util_double_same(dsend, 1.0));    /* Exactly */
  assert(util_double_same(drecv, dsend));  /* Exactly */

  isend[0] = -1;
  isend[1] = 0;
  isend[2] = +1;

  ireturn = MPI_Reduce(isend, irecv, 3, MPI_INT, MPI_SUM, 0, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(isend[0] == -1);
  assert(isend[1] == 0);
  assert(isend[2] == +1);
  assert(irecv[0] == -1);
  assert(irecv[1] == 0);
  assert(irecv[2] == +1);

  dvsend = (double *) malloc(2*sizeof(double));
  dvrecv = (double *) malloc(2*sizeof(double));

  assert(dvsend);
  assert(dvrecv);

  dvsend[0] = -1.0;
  dvsend[1] = +1.5;
  dvrecv[0] = 0.0;
  dvrecv[1] = 0.0;

  ireturn = MPI_Reduce(dvsend, dvrecv, 2, MPI_DOUBLE, MPI_SUM, 0, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(util_double_same(dvsend[0], -1.0));       /* All should be exact. */
  assert(util_double_same(dvsend[1], +1.5));
  assert(util_double_same(dvrecv[0], dvsend[0]));
  assert(util_double_same(dvrecv[1], dvsend[1]));

  free(dvsend);
  free(dvrecv);

  return ireturn;
}

/*****************************************************************************
 *
 *  test_mpi_allgather
 *
 *****************************************************************************/

int test_mpi_allgather(void) {

  int ireturn;
  double send[2];
  double recv[2];

  ireturn = MPI_Allgather(send, 2, MPI_DOUBLE, recv, 2, MPI_DOUBLE, comm_);

  assert(ireturn == MPI_SUCCESS);

  return ireturn;
}

/*****************************************************************************
 *
 *  test_mpi_type_contiguous
 *
 *****************************************************************************/

int test_mpi_type_contiguous(void) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;

  MPI_Type_contiguous(2, MPI_INT, &dt);
  MPI_Type_commit(&dt);
  assert(dt != MPI_DATATYPE_NULL);

  {
    MPI_Aint lb = -1;
    MPI_Aint extent = -1;

    MPI_Type_get_extent(dt, &lb, &extent);
    assert(lb == 0);
    assert(extent == 2*sizeof(int));
  }


  {
    /* MPI_Reduce(); something with a copy */
    int send[2] = {1, 2};
    int recv[2] = {0, 0};

    MPI_Reduce(send, recv, 1, dt, MPI_SUM, 0, MPI_COMM_WORLD);

    assert(send[0] == 1 && send[1] == 2);
    assert(recv[0] == 1 && recv[1] == 2);
  }

  MPI_Type_free(&dt);
  assert(dt == MPI_DATATYPE_NULL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_type_create_struct
 *
 *****************************************************************************/

int test_mpi_type_create_struct(void) {

  typedef struct test_s test_t;
  struct test_s {
    int a;
    double b;
  };

  MPI_Datatype dt = MPI_DATATYPE_NULL;

  {
    /* Commit */
    test_t data = {0};
    int count = 2;
    int blocklengths[2] = {1, 1};
    MPI_Aint displacements[3] = {0};
    MPI_Datatype datatypes[2] = {MPI_INT, MPI_DOUBLE};

    MPI_Get_address(&data,   displacements + 0);
    MPI_Get_address(&data.a, displacements + 1);
    MPI_Get_address(&data.b, displacements + 2);
    displacements[1] -= displacements[0];
    displacements[2] -= displacements[0];

    MPI_Type_create_struct(count, blocklengths, displacements + 1, datatypes,
			 &dt);
    MPI_Type_commit(&dt);
    assert(dt != MPI_DATATYPE_NULL);
  }

  {
    /* Extent */
    MPI_Aint lb = -1;
    MPI_Aint extent = -1;

    MPI_Type_get_extent(dt, &lb, &extent);
    assert(lb == 0);
    assert(extent == sizeof(test_t));
  }

  {
    /* Check can copy */
    test_t send = {.a = 1, .b = 2.0};
    test_t recv = {.a = 0, .b = 0.0};

    MPI_Reduce(&send, &recv, 1, dt, MPI_SUM, 0, MPI_COMM_WORLD);
    assert(send.a == recv.a);                     /* integer */
    assert(fabs(send.b - recv.b) < DBL_EPSILON);  /* double */
  }

  MPI_Type_free(&dt);
  assert(dt == MPI_DATATYPE_NULL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_op_create_function
 *  test_mpi_op_create
 *
 *****************************************************************************/

void test_op_create_function(void * invec, void * inoutvec, int * len,
			     MPI_Datatype * dt) {

  assert(invec);
  assert(inoutvec);
  assert(len);
  assert(dt);

  if (*dt == MPI_INT) { /* see comments below */
    int * myinoutvec = (int *) inoutvec;
    int * myinvec    = (int *) invec;
    for (int n = 0; n < *len; n++) {
      myinoutvec[n] = 2*myinvec[n];
    }
  }

  return;
}

static int test_mpi_op_create(void) {

  MPI_Op op = MPI_OP_NULL;

  MPI_Op_create((MPI_User_function *) test_op_create_function, 0, &op);
  assert(op != MPI_OP_NULL);

  {
    /* Smoke test; the implementation only ever copies; the function
     * is not called ... */
    int send = 1;
    int recv = 0;

    MPI_Reduce(&send, &recv, 1, MPI_INT, op, 0, MPI_COMM_WORLD);
    assert(recv == send);
  }

  MPI_Op_free(&op);
  assert(op == MPI_OP_NULL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_file_open
 *
 *****************************************************************************/

int test_mpi_file_open(void) {

  int ifail = MPI_SUCCESS;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  {
    /* fopen "r". We must have an existing file. */
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "/dev/null", MPI_MODE_RDONLY, info, &fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh != MPI_FILE_NULL);
    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh == MPI_FILE_NULL);
  }

  {
    /* fopen "w" */
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "zw.dat", MPI_MODE_WRONLY+MPI_MODE_CREATE,
			  info, &fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh != MPI_FILE_NULL);
    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh == MPI_FILE_NULL);
    unlink("zw.dat");
  }

  {
    /* fopen "a" */
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "z.dat",  MPI_MODE_WRONLY+MPI_MODE_APPEND,
			  info, &fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh != MPI_FILE_NULL);
    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh == MPI_FILE_NULL);
  }

  {
    /* fopen "r+" */
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "z.dat", MPI_MODE_RDWR, info, &fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh != MPI_FILE_NULL);
    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);
    assert(fh == MPI_FILE_NULL);
    unlink("z.dat");
  }

  /* Errors return when bad communicator */
  {
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(MPI_COMM_NULL, "/dev/null", MPI_MODE_RDWR, info,
			  &fh);
    assert(ifail == MPI_ERR_COMM);
  }

  /* Errors return when bad amode */
  {
    int amode = 0;
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "/dev/null", amode, info, &fh);
    assert(ifail == MPI_ERR_AMODE);
  }

  /* Errors return when no such file */
  {
    MPI_File fh = MPI_FILE_NULL;
    ifail = MPI_File_open(comm, "none-such", MPI_MODE_RDONLY, info, &fh);
    assert(ifail == MPI_ERR_NO_SUCH_FILE);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_file_get_view
 *
 *****************************************************************************/

static int test_mpi_file_get_view(void) {

  MPI_File fh = MPI_FILE_NULL;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  MPI_File_open(comm, "mpif.dat", MPI_MODE_WRONLY+MPI_MODE_CREATE, info, &fh);

  {
    MPI_Offset disp = 1;
    MPI_Datatype etype = MPI_DATATYPE_NULL;
    MPI_Datatype filetype = MPI_DATATYPE_NULL;
    char datarep[MPI_MAX_DATAREP_STRING] = {0};

    MPI_File_get_view(fh, &disp, &etype, &filetype, datarep);
    assert(disp == 0);
    assert(etype == MPI_BYTE);
    assert(filetype == MPI_BYTE);
    assert(strncmp(datarep, "native", MPI_MAX_DATAREP_STRING-1) == 0);
  }

  MPI_File_close(&fh);
  unlink("mpif.dat");

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_file_set_view
 *
 *****************************************************************************/

int test_mpi_file_set_view(void) {


  MPI_File fh = MPI_FILE_NULL;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  MPI_File_open(comm, "mpif.dat", MPI_MODE_WRONLY+MPI_MODE_CREATE, info, &fh);

  {
    /* The values here aren't really meaningful, but they will do ... */
    MPI_Offset disp = 1;
    MPI_Datatype etype = MPI_DOUBLE;
    MPI_Datatype filetype = MPI_INT;

    MPI_File_set_view(fh, disp, etype, filetype, "native", info);
  }

  {
    MPI_Offset disp = 0;
    MPI_Datatype etype = MPI_DATATYPE_NULL;
    MPI_Datatype filetype = MPI_DATATYPE_NULL;
    char datarep[MPI_MAX_DATAREP_STRING] = {0};

    MPI_File_get_view(fh, &disp, &etype, &filetype, datarep);
    assert(disp == 1);
    assert(etype == MPI_DOUBLE);
    assert(filetype == MPI_INT);
    assert(strncmp(datarep, "native", MPI_MAX_DATAREP_STRING-1) == 0);
  }

  MPI_File_close(&fh);
  unlink("mpif.dat");

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_type_create_subarray
 *
 *****************************************************************************/

int test_mpi_type_create_subarray(void) {

  {
    /* two dimensional (contiguous) */
    int ndims = 2;
    int sizes[2] = {128, 64};
    int subsizes[2] = {128, 64};
    int starts[2] = {0, 0};
    MPI_Datatype oldtype = MPI_CHAR;
    MPI_Datatype newtype = MPI_DATATYPE_NULL;

    MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C,
			     oldtype, &newtype);
    MPI_Type_commit(&newtype);
    assert(newtype != MPI_DATATYPE_NULL);
    MPI_Type_free(&newtype);
  }

  {
    /* three dimensional (contiguous) */
    int ndims = 3;
    int sizes[3] = {16, 8, 4};
    int subsizes[3] = {16, 8, 4};
    int starts[3] = {0, 0};
    MPI_Datatype oldtype = MPI_DOUBLE;
    MPI_Datatype newtype = MPI_DATATYPE_NULL;

    MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C,
			     oldtype, &newtype);
    MPI_Type_commit(&newtype);
    assert(newtype != MPI_DATATYPE_NULL);
    MPI_Type_free(&newtype);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_file_write_all
 *
 *  It is convenient to test MPI_File_read_all() at the same time.
 *
 *****************************************************************************/

int test_mpi_file_write_all(void) {

#define NX 23
#define NY 12

  int ifail = MPI_SUCCESS; /* return value */

  const char * filename = "mpi-file-write-all.dat";
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  /* Some very synthetic data */
  int ndims = 2;
  int sizes[2] = {NX, NY};
  int subsizes[2] = {NX, NY};
  int starts[2] = {0, 0};

  /* Create a subarray type */

  MPI_Datatype etype = MPI_DOUBLE;
  MPI_Datatype filetype = MPI_DATATYPE_NULL;

  MPI_Type_create_subarray(ndims, sizes, subsizes, starts, MPI_ORDER_C,
			     etype, &filetype);
  MPI_Type_commit(&filetype);

  {
    /* Write */

    MPI_File fh = MPI_FILE_NULL;
    MPI_Offset disp = 0;

    int count = 1;
    double wbuf[NX*NY] = {0};

    /* Some test values */
    for (int id = 0; id < NX*NY; id++) {
      wbuf[id] = 1.0*id;
    }

    ifail = MPI_File_open(comm, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE,
			  info, &fh);
    assert(ifail == MPI_SUCCESS);

    /* Set the view */
    /* As this is serial the datetype is the filetype */

    ifail = MPI_File_set_view(fh, disp, etype, filetype, "native", info);
    assert(ifail == MPI_SUCCESS);

    ifail = MPI_File_write_all(fh, wbuf, count, filetype, MPI_STATUS_IGNORE);
    assert(ifail == MPI_SUCCESS);

    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);
  }


  {
    /* Re-read the same file */
    MPI_File fh = MPI_FILE_NULL;
    MPI_Offset disp = 0;

    int ifail = MPI_SUCCESS;
    int count = 1;
    double rbuf[NX*NY] = {0};

    ifail = MPI_File_open(comm, filename, MPI_MODE_RDONLY, info, &fh);
    assert(ifail == MPI_SUCCESS);

    /* Set the view */
    ifail = MPI_File_set_view(fh, disp, etype, filetype, "native", info);
    assert(ifail == MPI_SUCCESS);

    ifail = MPI_File_read_all(fh, rbuf, count, filetype, MPI_STATUS_IGNORE);
    assert(ifail == MPI_SUCCESS);

    ifail = MPI_File_close(&fh);
    assert(ifail == MPI_SUCCESS);

    for (int id = 0; id < NX*NY; id++) {
      assert(fabs(rbuf[id] - 1.0*id) < DBL_EPSILON);
      if (fabs(rbuf[id] - 1.0*id) > DBL_EPSILON) ifail += 1;
    }
  }

  MPI_Type_free(&filetype);
  unlink(filename);

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_comm_split_type
 *
 *****************************************************************************/

int test_mpi_comm_split_type(void) {

  MPI_Comm comm = MPI_COMM_NULL;
  int key = 0;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL,
		      &comm);
  MPI_Comm_free(&comm);

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_cart_create
 *
 *****************************************************************************/

static int test_mpi_cart_create(void) {

  int ifail = 0;

  {
    int ndims = 2;
    int dims[2]  = {1, 1};
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm comm = MPI_COMM_NULL;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm);

    assert(comm != MPI_COMM_NULL);
    assert(comm != MPI_COMM_SELF);

    MPI_Comm_free(&comm);
  }

  {
    int ndims = 3;
    int dims[3] = {1, 1, 1};
    int periods[3] = {1, 0, 1};
    int reorder = 1;
    MPI_Comm comm = MPI_COMM_NULL;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm);

    assert(comm != MPI_COMM_NULL);
    assert(comm != MPI_COMM_SELF);

    MPI_Comm_free(&comm);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_cart_get
 *
 *****************************************************************************/

static int test_mpi_cart_get(void) {

  int ifail = 0;

  {
    int ndims = 2;
    int dims[2]  = {1, 1};
    int periods[2] = {0, 0};
    int reorder = 0;
    MPI_Comm comm = MPI_COMM_NULL;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm);

    {
      int xdims[2] = {0};
      int xperiods[2] = {-1, -1};
      int coords[2]   = {-1, -1};
      MPI_Cart_get(comm, ndims, xdims, xperiods, coords);

      assert(xdims[0]    == dims[0]);
      assert(xdims[1]    == dims[1]);
      assert(xperiods[0] == periods[0]);
      assert(xperiods[1] == periods[1]);
      assert(coords[0]   == 0);
      assert(coords[1]   == 0);
    }

    MPI_Comm_free(&comm);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_dims_create
 *
 *****************************************************************************/

static int test_mpi_dims_create(void) {

  int ifail = 0;

  {
    int nnodes = 1;
    int ndims  = 2;
    int dims[2] = {0, 0};

    MPI_Dims_create(nnodes, ndims, dims);

    assert(dims[0] == 1);
    assert(dims[1] == 1);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_mpi_type_size
 *
 *****************************************************************************/

static int test_mpi_type_size(void) {

  /* Intrinsic */
  {
    int sz = -1;
    MPI_Type_size(MPI_DOUBLE, &sz);
    assert(sz == sizeof(double));
  }

  /* User */
  {
    int sz = -1;
    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_vector(2, 3, 0, MPI_INT, &dt);
    MPI_Type_commit(&dt);
    MPI_Type_size(dt, &sz);
    assert(sz == 2*3*sizeof(int));
    MPI_Type_free(&dt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_mpi_type_get_extent
 *
 *****************************************************************************/

static int test_mpi_type_get_extent(void) {

  /* Intrinsic */
  {
    MPI_Aint lb = 999;
    MPI_Aint extent = 999;
    MPI_Type_get_extent(MPI_INT, &lb, &extent);
    assert(lb == 0);
    assert(extent == sizeof(int));
  }

  /* User */
  {
    MPI_Datatype udt = MPI_DATATYPE_NULL;
    MPI_Aint lb = 999;
    MPI_Aint extent = 999;

    MPI_Type_contiguous(3, MPI_INT, &udt);
    MPI_Type_commit(&udt);
    MPI_Type_get_extent(udt, &lb, &extent);
    assert(lb == 0);
    assert(extent == 3*sizeof(int));
    MPI_Type_free(&udt);
  }

  return 0;
}
