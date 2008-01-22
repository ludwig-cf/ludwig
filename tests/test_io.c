/*****************************************************************************
 *
 *  io_test
 *
 *  Test code for the lattice I/O harness code.
 *
 *  $Id: test_io.c,v 1.1 2008-01-22 14:49:34 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "io_harness.h"

static void test_io_info_struct(void);
static void test_processor_independent(void);
static int  test_read_1(FILE *, const int, const int, const int);
static int  test_write_1(FILE *, const int, const int, const int);
static int  test_read_2(FILE *, const int, const int, const int);
static int  test_write_2(FILE *, const int, const int, const int);

static double test_index(int, int, int);

int main (int argc, char ** argv) {

  /* Take default system size */

  pe_init(argc, argv);
  coords_init();

  test_io_info_struct();
  test_processor_independent();

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
  int ifail;

  sprintf(stub, "%s", "ztest_file");

  info("\nTesting io info struct...\n");
  info("Allocating one io_info object...");

  io_info = io_info_create();
  assert(io_info != (struct io_info_t *) NULL);
  info("ok\n");

  io_info_set_name(io_info, "Test double data");
  io_info_set_bytesize(io_info, sizeof(double));
  io_info_set_write(io_info, test_write_1);
  io_info_set_read(io_info, test_read_1);

  info("Testing write to filename stub %s ...", stub);
  io_write(stub, io_info);
  info("ok\n");

  info("Re-read test data...\n");
  io_read(stub, io_info);

  info("Release io_info struct...");
  io_info_destroy(io_info);
  info("ok\n");

  /* Assume 1 I/O group */

  info("Removing data file %s...", stub);
  ifail = remove(stub);
  assert(ifail == 0);
  info("ok\n\n");

  sprintf(stub, "%s", "ztest_file.meta"); 
  info("Removing metadata file %s...", stub);
  ifail = remove(stub);
  assert(ifail == 0);
  info("ok\n\n");

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
  int ifail;
  int grid[3] = {1, 1, 1};

  sprintf(stub, "%s", "ztest_file");

  info("\nTesting processor indendent read/write\n");
  info("Allocating one io_info object...");

  io_info = io_info_create_with_grid(grid);
  assert(io_info != (struct io_info_t *) NULL);
  info("ok\n");

  io_info_set_name(io_info, "Test int data");
  io_info_set_bytesize(io_info, sizeof(double));
  io_info_set_write(io_info, test_write_2);
  io_info_set_read(io_info, test_read_2);

  info("Setting decomposition independent true...ok\n");
  io_info_set_processor_independent(io_info);

  info("Write integer data... ");
  io_write(stub, io_info);
  info("ok\n");

  info("Re-read test data...\n");
  io_read(stub, io_info);

  info("Release io_info struct...");
  io_info_destroy(io_info);
  info("ok\n");

  /* Assume 1 I/O group */

  info("Removing data file %s...", stub);
  ifail = remove(stub);
  assert(ifail == 0);
  info("ok\n\n");

  sprintf(stub, "%s", "ztest_file.meta"); 
  info("Removing metadata file %s...", stub);
  ifail = remove(stub);
  assert(ifail == 0);
  info("ok\n\n");

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
  assert (n == 1);

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
  assert(n == 1);

  if (fabs(dref - dindex) > 0.5) {
    verbose("test_read_1 failed at %d %d %d\n", ic, jc, kc);
    verbose("Expecting %f but read %f\n", dref, dindex);
  }

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
  assert(n == 1);

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
  assert(n == 1);

  n = (int) test_index(ic, jc, kc);
  assert(index == n);

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

  get_N_offset(noffset);

  ic += noffset[X];
  jc += noffset[Y];
  kc += noffset[Z];

  dindex = L(Y)*L(Z)*ic + L(Z)*jc + 1.0*kc;

  return dindex;
}
