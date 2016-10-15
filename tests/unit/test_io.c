/*****************************************************************************
 *
 *  test_io
 *
 *  Test code for the lattice I/O harness code.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
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
  /* if (pe_size() == cart_size(X)) test_processor_independent();
     test_ascii();*/

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
  io_info_arg_t args;
  io_info_t * io_info = NULL;

  assert(pe);
  assert(cs);

  sprintf(stubp, "/tmp/temp-test-io-file");

  /*
  info("\nTesting io info struct...\n");
  info("Allocating one io_info object...");
  */

  args.grid[X] = 1;
  args.grid[Y] = 1;
  args.grid[Z] = 1;

  io_info_create(pe, cs, &args, &io_info);
  assert(io_info);

  /* info("Address of write function %p\n", test_write_1);*/

  io_info_set_name(io_info, "Test double data");
  io_info_set_bytesize(io_info, sizeof(double));
  io_info_write_set(io_info, IO_FORMAT_BINARY, test_io_write1);
  io_info_read_set(io_info, IO_FORMAT_BINARY, test_io_read1);

  io_info_format_set(io_info, IO_FORMAT_BINARY, IO_FORMAT_BINARY);
  io_info_set_processor_dependent(io_info);

  /* info("Testing write to filename stub %s ...", stub);*/

  io_write_data(io_info, stubp, &data);
  MPI_Barrier(MPI_COMM_WORLD);

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
  assert(n == 1);

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
  assert(n == 1);
  assert(fabs(data - s->dref*index) < DBL_EPSILON);

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
  assert(indata == data->iref*index);

  return n;
}

/*****************************************************************************
 *
 *  test_write_3
 *
 *  ASCII write (int data)
 *
 *****************************************************************************/

int test_io_write3(FILE * fp, int index, void * self) {

  int n;
  test_io_t * data = (test_io_t *) self;

  n = fprintf(fp, "%d\n", index*data->iref);
  assert(n >= 2);

  return n;
}
