/*****************************************************************************
 *
 *  io_test
 *
 *  Test code for the lattice I/O harness code.
 *
 *  $Id: test_io.c,v 1.4 2010-11-02 17:51:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "io_harness.h"
#include "runtime.h"
#include "tests.h"

static void test_io_info_struct(void);
static void test_processor_independent(void);
static void test_ascii(void);
static int  test_read_1(FILE *, const int, const int, const int);
static int  test_write_1(FILE *, const int, const int, const int);
static int  test_read_2(FILE *, const int, const int, const int);
static int  test_write_2(FILE *, const int, const int, const int);
static int  test_read_3(FILE *, const int, const int, const int);
static int  test_write_3(FILE *, const int, const int, const int);

static double test_index(int, int, int);

int main (int argc, char ** argv) {

  /* Take default system size */

  pe_init(argc, argv);
  if (argc > 1) RUN_read_input_file(argv[1]);
  coords_init();

  test_io_info_struct();
  if (pe_size() == cart_size(X)) test_processor_independent();
  test_ascii();

  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_io_info_struct
 *
 *****************************************************************************/

void test_io_info_struct() {

  struct io_info_t * io_info;
  char stub[FILENAME_MAX];

  sprintf(stub, "%s", "ztest_file");

  info("\nTesting io info struct...\n");
  info("Allocating one io_info object...");

  io_info = io_info_create();
  test_assert(io_info != (struct io_info_t *) NULL);
  info("ok\n");

  info("Address of write function %p\n", test_write_1);

  io_info_set_name(io_info, "Test double data");
  io_info_set_bytesize(io_info, sizeof(double));
  io_info_set_write(io_info, test_write_1);
  io_info_set_read(io_info, test_read_1);
  io_info_set_processor_dependent(io_info);

  info("Testing write to filename stub %s ...", stub);
  io_write(stub, io_info);

  MPI_Barrier(MPI_COMM_WORLD);
  info("ok\n");

  info("Re-read test data...\n");
  io_read(stub, io_info);

  MPI_Barrier(MPI_COMM_WORLD);

  info("Release io_info struct...");
  io_info_destroy(io_info);
  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_processor_independent
 *
 *****************************************************************************/

void test_processor_independent() {

  struct io_info_t * io_info;
  char stub[FILENAME_MAX];
  int grid[3] = {1, 1, 1};

  sprintf(stub, "%s", "ztest_file");

  info("\nTesting processor indendent read/write\n");
  info("Allocating one io_info object...");

  io_info = io_info_create_with_grid(grid);
  test_assert(io_info != (struct io_info_t *) NULL);
  info("ok\n");

  io_info_set_name(io_info, "Test int data");
  io_info_set_bytesize(io_info, sizeof(int));
  io_info_set_write(io_info, test_write_2);
  io_info_set_read(io_info, test_read_2);

  info("Setting decomposition independent true...ok\n");
  io_info_set_processor_independent(io_info);

  info("Write integer data... ");
  io_write(stub, io_info);
  info("ok\n");

  MPI_Barrier(MPI_COMM_WORLD);

  info("Re-read test data...\n");
  io_read(stub, io_info);
  info("ok\n");

  MPI_Barrier(MPI_COMM_WORLD);

  info("Release io_info struct...");
  io_info_destroy(io_info);
  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_ascii
 *
 *****************************************************************************/

void test_ascii() {

  struct io_info_t * io_info;
  char filestub[FILENAME_MAX];
  int grid[3] = {1, 1, 1};

  info("Switching to ASCII format...\n");

  io_info = io_info_create_with_grid(grid);

  io_info_set_write_ascii(io_info, test_write_3);
  io_info_set_read_ascii(io_info, test_read_3);
  io_info_set_processor_dependent(io_info);
  io_info_set_format_ascii(io_info);

  sprintf(filestub, "ztest_ascii");

  info("ASCII write...\n");
  io_write(filestub, io_info);
  info("ASCII write ok\n");

  info("ASCII read...\n");
  io_read(filestub, io_info);
  info("ASCII read ok\n");

  io_info_destroy(io_info);

  return;
}

/*****************************************************************************
 *
 *  test_write_1
 *
 *****************************************************************************/

static int test_write_1(FILE * fp, const int ic, const int jc, const int kc) {

  double dindex;
  int    n;

  dindex = test_index(ic, jc, kc);

  n = fwrite(&dindex, sizeof(double), 1, fp);
  test_assert (n == 1);

  return n;
}

/*****************************************************************************
 *
 *  test_read_1
 *
 *****************************************************************************/

static int test_read_1(FILE * fp, const int ic, const int jc, const int kc) {

  double dindex, dref;
  int    n;

  dref = test_index(ic, jc, kc);

  n = fread(&dindex, sizeof(double), 1, fp);
  test_assert(n == 1);

  if (fabs(dref - dindex) > 0.5) {
    verbose("test_read_1 failed at %d %d %d\n", ic, jc, kc);
    verbose("Expecting %f but read %f\n", dref, dindex);
  }
  test_assert(fabs(dref - dindex) < TEST_DOUBLE_TOLERANCE);

  if (jc == 1 && kc == 1 && ic % 8 == 0) {
    verbose("Test read ic = %d ok\n", ic);
  }

  return n;
}

/*****************************************************************************
 *
 *  test_write_2
 *
 *  Integer write (global index)
 *
 *****************************************************************************/

int test_write_2(FILE * fp, const int ic, const int jc, const int kc) {

  int index, n;

  index = test_index(ic, jc, kc);

  n = fwrite(&index, sizeof(int), 1, fp);
  test_assert(n == 1);

  return n;
}

/*****************************************************************************
 *
 *  test_read_2
 *
 *  Read integer (global index)
 *
 *****************************************************************************/

int test_read_2(FILE * fp, const int ic, const int jc, const int kc) {

  int index,  n;

  n = fread(&index, sizeof(int), 1, fp);
  test_assert(n == 1);

  n = (int) test_index(ic, jc, kc);

  if (n != index) {
    verbose("test_read_2 (%d,%d,%d) read %d expected %d\n", ic,jc,kc,index,n);
  }

  test_assert(index == n);

  return n;
}

/*****************************************************************************
 *
 *  test_read_3
 *
 *  ASCII read (char data)
 *
 *****************************************************************************/

int test_read_3(FILE * fp, const int ic, const int jc, const int kc) {

  int n;
  int indata;

  n = fscanf(fp, "%d\n", &indata);

  test_assert(n == 1);
  test_assert(indata == (ic + jc + kc));

  return n;
}

/*****************************************************************************
 *
 *  test_write_3
 *
 *  ASCII write (int data)
 *
 *****************************************************************************/

int test_write_3(FILE * fp, const int ic, const int jc, const int kc) {

  int n;

  n = fprintf(fp, "%d\n", (ic + jc + kc));
  test_assert(n >= 2);

  return n;
}

/*****************************************************************************
 *
 *  test_index
 *
 *****************************************************************************/

double test_index(int ic, int jc, int kc) {

  double dindex;
  int    noffset[3];

  coords_nlocal_offset(noffset);

  ic += noffset[X];
  jc += noffset[Y];
  kc += noffset[Z];

  dindex = L(Y)*L(Z)*ic + L(Z)*jc + 1.0*kc;

  return dindex;
}
