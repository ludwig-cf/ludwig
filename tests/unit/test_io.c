/*****************************************************************************
 *
 *  test_io
 *
 *  Test code for the lattice I/O harness code.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "io_harness.h"
#include "tests.h"

typedef struct test_io_s test_io_t;
struct test_io_s {
  int iref;
  double dref;
};

int do_test_io_info_struct(pe_t * pe, cs_t * cs);
static int  test_io_read1(FILE *, int index, void * self);
static int  test_io_write1(FILE *, int index, void * self);
static int  test_io_read3(FILE *, int index, void * self);
static int  test_io_write3(FILE *, int index, void * self);

/*****************************************************************************
 *
 *  test_io_suite
 *
 *****************************************************************************/

int test_io_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  
  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  do_test_io_info_struct(pe, cs);

  pe_info(pe, "PASS     ./unit/test_io\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_info_struct
 *
 *****************************************************************************/

int do_test_io_info_struct(pe_t * pe, cs_t * cs) {

  char stubp[FILENAME_MAX];
  test_io_t data = {2, 1.0};
  io_info_args_t args;
  io_info_t * io_info = NULL;

  assert(pe);
  assert(cs);

  sprintf(stubp, "/tmp/temp-test-io-file");

  args.grid[X] = 1;
  args.grid[Y] = 1;
  args.grid[Z] = 1;

  io_info_create(pe, cs, &args, &io_info);
  assert(io_info);

  io_info_set_name(io_info, "Test double data");
  io_info_set_bytesize(io_info, IO_FORMAT_BINARY, sizeof(double));
  io_info_set_bytesize(io_info, IO_FORMAT_ASCII, 10); /* See below */
  io_info_write_set(io_info, IO_FORMAT_BINARY, test_io_write1);
  io_info_read_set(io_info, IO_FORMAT_BINARY, test_io_read1);

  io_info_format_set(io_info, IO_FORMAT_BINARY, IO_FORMAT_BINARY);
  io_info_set_processor_dependent(io_info);

  io_write_data(io_info, stubp, &data);
  MPI_Barrier(MPI_COMM_WORLD);

  io_info_format_in_set(io_info, IO_FORMAT_BINARY_SERIAL);
  io_read_data(io_info, stubp, &data);
  MPI_Barrier(MPI_COMM_WORLD);

  io_remove((const char *) stubp, io_info);
  MPI_Barrier(MPI_COMM_WORLD);

  /* ASCII */

  io_info_read_set(io_info, IO_FORMAT_ASCII, test_io_read3);
  io_info_write_set(io_info, IO_FORMAT_ASCII, test_io_write3);
  io_info_format_set(io_info, IO_FORMAT_ASCII, IO_FORMAT_ASCII);

  io_write_data(io_info, stubp, &data);
  MPI_Barrier(MPI_COMM_WORLD);

  io_info_format_in_set(io_info, IO_FORMAT_ASCII_SERIAL);
  io_read_data(io_info, stubp, &data);
  MPI_Barrier(MPI_COMM_WORLD);

  io_remove(stubp, io_info);

  /* Meta data */

  io_write_metadata_file(io_info, stubp);
  MPI_Barrier(MPI_COMM_WORLD);

  io_remove_metadata(io_info, stubp);

  io_info_free(io_info);

  return 0;
}

/*****************************************************************************
 *
 *  test_write_1
 *
 *****************************************************************************/

static int test_io_write1(FILE * fp, int index, void * self) {

  int n;
  double data;
  test_io_t * s = (test_io_t *) self;

  assert(fp);

  data = s->dref*index;
  n = fwrite(&data, sizeof(double), 1, fp);
  test_assert(n == 1);

  return n;
}

/*****************************************************************************
 *
 *  test_read_1
 *
 *****************************************************************************/

static int test_io_read1(FILE * fp, int index, void * self) {

  int n;
  double data;
  test_io_t * s = (test_io_t *) self;

  assert(fp);
  assert(s);

  n = fread(&data, sizeof(double), 1, fp);
  test_assert(n == 1);
  test_assert(fabs(data - s->dref*index) < DBL_EPSILON);

  return n;
}

/*****************************************************************************
 *
 *  test_read_3
 *
 *  ASCII read (char data)
 *
 *****************************************************************************/

int test_io_read3(FILE * fp, int index, void * self) {

  int n;
  int indata;
  test_io_t * data = (test_io_t *) self;

  n = fscanf(fp, "%d\n", &indata);

  assert(n == 1);
  if (n == 1) test_assert(indata == data->iref*index);

  return n;
}

/*****************************************************************************
 *
 *  test_write_3
 *
 *  ASCII write (int data) Must have a fixed format.
 *
 *****************************************************************************/

int test_io_write3(FILE * fp, int index, void * self) {

  int n;
  test_io_t * data = (test_io_t *) self;

  n = fprintf(fp, "%9d\n", index*data->iref);
  test_assert(n == 10);

  return n;
}
