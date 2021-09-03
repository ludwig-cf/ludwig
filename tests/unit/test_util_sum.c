/*****************************************************************************
 *
 *  test_util_sum.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "util_sum.h"
#include "tests.h"

int test_kahan_zero(pe_t * pe);
int test_kahan_sum(pe_t * pe);
int test_kahan_add_double(pe_t * pe);
int test_kahan_mpi_datatype(pe_t * pe);
int test_kahan_mpi_op_sum(pe_t * pe);

int test_klein_zero(pe_t * pe);
int test_klein_sum(pe_t * pe);
int test_klein_add_double(pe_t * pe);
int test_klein_mpi_datatype(pe_t * pe);
int test_klein_mpi_op_sum(pe_t * pe);

/* "Private" functions in util_sum.c */

void kahan_mpi_op_sum_function(kahan_t * invec, kahan_t * inoutvec, int * len,
			       MPI_Datatype * dt);
void klein_mpi_op_sum_function(klein_t * invec, klein_t * inoutvec, int * len,
			       MPI_Datatype * dt);

/*****************************************************************************
 *
 *  test_util_sum_suite
 *
 *****************************************************************************/

int test_util_sum_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_kahan_zero(pe);
  test_kahan_sum(pe);
  test_kahan_add_double(pe);
  test_kahan_mpi_datatype(pe);
  test_kahan_mpi_op_sum(pe);

  test_klein_zero(pe);
  test_klein_sum(pe);
  test_klein_add_double(pe);
  test_klein_mpi_datatype(pe);
  test_klein_mpi_op_sum(pe);

  pe_info(pe, "PASS     ./unit/test_util_sum\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_kahan_zero
 *
 *****************************************************************************/

int test_kahan_zero(pe_t * pe) {

  kahan_t k = kahan_zero();

  assert(pe);

  assert(k.lock == 0);
  assert(k.sum  == 0.0); /* Exact */
  assert(k.cs   == 0.0); /* Exact */

  return k.lock;
}

/*****************************************************************************
 *
 *  test_klein_zero
 *
 *****************************************************************************/

int test_klein_zero(pe_t * pe) {

  klein_t k = klein_zero();

  assert(pe);

  assert(k.lock == 0);
  assert(k.sum == 0.0); /* Exact */
  assert(k.cs  == 0.0); /* Exact */
  assert(k.ccs == 0.0); /* Exact */

  return k.lock;
}

/*****************************************************************************
 *
 *  test_kahan_sum
 *
 *****************************************************************************/

int test_kahan_sum(pe_t * pe) {

  kahan_t k = kahan_zero();

  assert(pe);

  {
    k.sum = 1.0;
    k.cs  = 2.0;
    assert(fabs(kahan_sum(&k) - 3.0) < DBL_EPSILON);
  }

  return k.lock;
}

/*****************************************************************************
 *
 *  test_kahan_add_double
 *
 *****************************************************************************/

int test_kahan_add_double(pe_t * pe) {

  kahan_t k = kahan_zero();
  double a = 1.0;
  double b = 1.0e-17;

  assert(pe);

  /* Note: here must be add a then add b; not add b then add a */
  kahan_add_double(&k, a);
  kahan_add_double(&k, b);
  assert(fabs(kahan_sum(&k) - a) < DBL_EPSILON);

  kahan_add_double(&k, -b);
  kahan_add_double(&k, -a);
  assert(kahan_sum(&k) == 0.0);

  return k.lock;
}

/*****************************************************************************
 *
 *  test_kahan_mpi_datatype
 *
 *****************************************************************************/

int test_kahan_mpi_datatype(pe_t * pe) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;

  assert(pe);

  kahan_mpi_datatype(&dt);

  assert(dt != MPI_DATATYPE_NULL);

  {
    MPI_Aint lb = -1;
    MPI_Aint extent = -1;

    MPI_Type_get_extent(dt, &lb, &extent);
    assert(lb == 0);
    assert(extent == sizeof(kahan_t));
  }

  {
    /* Bcast as example */
    kahan_t k = kahan_zero();
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;

    pe_mpi_comm(pe, &comm);
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) kahan_add_double(&k, 1.0);
    MPI_Bcast(&k, 1, dt, 0, comm);
    assert(fabs(kahan_sum(&k) - 1.0) < DBL_EPSILON);
  }

  MPI_Type_free(&dt);

  return 0;
}

/*****************************************************************************
 *
 *  test_kahan_mpi_op_sum
 *
 *****************************************************************************/

int test_kahan_mpi_op_sum(pe_t * pe) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Op op = MPI_OP_NULL;

  assert(pe);

  kahan_mpi_datatype(&dt);
  kahan_mpi_op_sum(&op);
  assert(op != MPI_OP_NULL);

  {
    /* internal smoke test */

    kahan_t invec = kahan_zero();
    kahan_t inoutvec = kahan_zero();
    int count = 1;

    kahan_mpi_op_sum_function(&invec, &inoutvec, &count, &dt);
  }

  {
    /* Allreduce as example */
    kahan_t send = kahan_zero();
    kahan_t recv = kahan_zero();
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;

    pe_mpi_comm(pe, &comm);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) kahan_add_double(&send, 1.0);

    MPI_Allreduce(&send, &recv, 1, dt, op, comm);
    assert(fabs(kahan_sum(&recv) - 1.0) < DBL_EPSILON);
  }

  MPI_Op_free(&op);
  MPI_Type_free(&dt);

  return 0;
}

/*****************************************************************************
 *
 * test_klein_sum
 *
 *****************************************************************************/

int test_klein_sum(pe_t * pe) {

  klein_t k = klein_zero();

  assert(pe);

  {
    k.sum = 1.0;
    k.cs  = 2.0;
    k.ccs = 3.0;
    assert(fabs(klein_sum(&k) - 6.0) < DBL_EPSILON);
  }

  return k.lock;
}

/*****************************************************************************
 *
 *  test_klein_add_double
 *
 *****************************************************************************/

int test_klein_add_double(pe_t * pe) {

  klein_t k = klein_zero();
  const double a = 1.0;
  const double b = 1.0e-17;
  const double c = 1.0e-34;

  assert(pe);

  klein_add_double(&k, c);
  klein_add_double(&k, b);
  klein_add_double(&k, a);
  assert(klein_sum(&k) == a);   /* These should come out exact */

  klein_add_double(&k, -a);
  assert(klein_sum(&k) == b);
  klein_add_double(&k, -b);
  assert(klein_sum(&k) == c);
  klein_add_double(&k, -c);
  assert(klein_sum(&k) == 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  test_klein_mpi_datatype
 *
 *****************************************************************************/

int test_klein_mpi_datatype(pe_t * pe) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;

  assert(pe);

  klein_mpi_datatype(&dt);

  assert(dt != MPI_DATATYPE_NULL);

  {
    MPI_Aint lb = -1;
    MPI_Aint extent = -1;

    MPI_Type_get_extent(dt, &lb, &extent);
    assert(lb == 0);
    assert(extent == sizeof(klein_t));
  }

  {
    /* Bcast as example */
    klein_t k = klein_zero();
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;

    pe_mpi_comm(pe, &comm);
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) klein_add_double(&k, 1.0);
    MPI_Bcast(&k, 1, dt, 0, comm);
    assert(fabs(klein_sum(&k) - 1.0) < DBL_EPSILON);
  }

  MPI_Type_free(&dt);

  return 0;
}

/*****************************************************************************
 *
 *  test_klein_mpi_op_sum
 *
 *****************************************************************************/

int test_klein_mpi_op_sum(pe_t * pe) {

  MPI_Datatype dt = MPI_DATATYPE_NULL;
  MPI_Op op = MPI_OP_NULL;

  assert(pe);

  klein_mpi_datatype(&dt);
  klein_mpi_op_sum(&op);
  assert(op != MPI_OP_NULL);

  {
    /* internal smoke test */

    klein_t invec = klein_zero();
    klein_t inoutvec = klein_zero();
    int count = 1;

    klein_mpi_op_sum_function(&invec, &inoutvec, &count, &dt);
  }

  {
    /* Allreduce as example */
    klein_t send = klein_zero();
    klein_t recv = klein_zero();
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;

    pe_mpi_comm(pe, &comm);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) klein_add_double(&send, 1.0);

    MPI_Allreduce(&send, &recv, 1, dt, op, comm);
    assert(fabs(klein_sum(&recv) - 1.0) < DBL_EPSILON);
  }

  MPI_Op_free(&op);
  MPI_Type_free(&dt);

  return 0;
}
