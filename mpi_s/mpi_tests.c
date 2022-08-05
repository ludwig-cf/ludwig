/*****************************************************************************
 *
 *  mpi_tests
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
static int test_mpi_op_create(void);
static int test_mpi_file_open(void);

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
  test_mpi_op_create();

  test_mpi_file_open();

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

  int ireturn;
  double dsend, drecv;
  int isend[3], irecv[3];

  dsend = 1.0; drecv = 0.0;
  ireturn = MPI_Allreduce(&dsend, &drecv, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(dsend == 1.0);   /* Exactly */
  assert(drecv == dsend); /* Exactly */

  isend[0] = -1;
  isend[1] = 0;
  isend[2] = +1;

  ireturn = MPI_Allreduce(isend, irecv, 3, MPI_INT, MPI_SUM, comm_);
  assert(ireturn == MPI_SUCCESS);
  assert(isend[0] == -1);
  assert(isend[1] == 0);
  assert(isend[2] == +1);
  assert(irecv[0] == -1);
  assert(irecv[1] == 0);
  assert(irecv[2] == +1);

  return ireturn;
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
  assert(dsend == 1.0);    /* Exactly */
  assert(drecv == dsend);  /* Exactly */

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
  assert(dvsend[0] == -1.0);       /* All should be exact. */
  assert(dvsend[1] == +1.5);
  assert(dvrecv[0] == dvsend[0]);
  assert(dvrecv[1] == dvsend[1]);

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
    assert(send.a == recv.a);
    assert(fabs(send.b - recv.b) < DBL_EPSILON);
  }

  MPI_Type_free(&dt);
  assert(dt == MPI_DATATYPE_NULL);

  return MPI_SUCCESS;
}

/*****************************************************************************
 *
 *  test_mpi_op_create
 *
 *****************************************************************************/

void test_op_create_function(void * invec, void * inoutvec, int * len,
			     MPI_Datatype * dt) {

  assert(invec);
  assert(inoutvec);
  assert(len);
  assert(dt);

  return;
}

static int test_mpi_op_create(void) {

  MPI_Op op = MPI_OP_NULL;

  MPI_Op_create((MPI_User_function *) test_op_create_function, 0, &op);
  assert(op != MPI_OP_NULL);

  {
    /* Smoke test */
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

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Info info = MPI_INFO_NULL;

  {
    /* fopen "r". We must have an existing file. */
    MPI_File fh = MPI_FILE_NULL;
    MPI_File_open(comm, "/dev/null", MPI_MODE_RDONLY, info, &fh);
    assert(fh != MPI_FILE_NULL);
    MPI_File_close(&fh);
    assert(fh == MPI_FILE_NULL);
  }

  {
    /* fopen "w" */
    MPI_File fh = MPI_FILE_NULL;
    MPI_File_open(comm, "zw.dat", MPI_MODE_WRONLY+MPI_MODE_CREATE, info, &fh);
    assert(fh != MPI_FILE_NULL);
    MPI_File_close(&fh);
    assert(fh == MPI_FILE_NULL);
    unlink("zw.dat");
  }

  {
    /* fopen "a" */
    MPI_File fh = MPI_FILE_NULL;
    MPI_File_open(comm, "z.dat",  MPI_MODE_WRONLY+MPI_MODE_APPEND, info, &fh);
    assert(fh != MPI_FILE_NULL);
    MPI_File_close(&fh);
    assert(fh == MPI_FILE_NULL);
  }

  {
    /* fopen "r+" */
    MPI_File fh = MPI_FILE_NULL;
    MPI_File_open(comm, "z.dat", MPI_MODE_RDWR, info, &fh);
    assert(fh != MPI_FILE_NULL);
    MPI_File_close(&fh);
    assert(fh == MPI_FILE_NULL);
    unlink("z.dat");
  }

  return 0;
}
