/*****************************************************************************
 *
 *  test_field.c
 *
 *  Unit test for field structure.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "field_s.h"

#include "test_coords_field.h"
#include "tests.h"

static int do_test0(void);
static int do_test1(void);
static int do_test3(void);
static int do_test5(void);
static int do_test_io(int nf, int io_format);
static int test_field_halo(field_t * phi);

/*****************************************************************************
 *
 *  test_field_suite
 *
 *****************************************************************************/

int test_field_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* info("\nOrder parameter tests...\n");*/

  do_test0();
  do_test1();
  do_test3();
  do_test5();

  do_test_io(1, IO_FORMAT_ASCII);
  do_test_io(1, IO_FORMAT_BINARY);
  do_test_io(5, IO_FORMAT_ASCII);
  do_test_io(5, IO_FORMAT_BINARY);

  pe_info(pe, "PASS     ./unit/test_field\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test0
 *
 *  Small system test.
 *
 *****************************************************************************/

static int do_test0(void) {

  int nfref = 1;
  int nhalo = 2;
  int ntotal[3] = {8, 8, 8};
  field_t * phi = NULL;

  
  coords_nhalo_set(nhalo);
  coords_ntotal_set(ntotal);
  coords_init();
  le_init();

  field_create(nfref, "phi", &phi);
  field_init(phi, nhalo);

  /* Halo */
  test_field_halo(phi);

  field_free(phi);
  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *  Scalar order parameter.
 *
 *****************************************************************************/

int do_test1(void) {

  int nfref = 1;
  int nf;
  int nhalo = 2;
  int index = 1;
  double ref;
  double value;
  field_t * phi = NULL;

  coords_nhalo_set(nhalo);
  coords_init();
  le_init();

  field_create(nfref, "phi", &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  field_init(phi, nhalo);

  ref = 1.0;
  field_scalar_set(phi, index, ref);
  field_scalar(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  ref = -1.0;
  field_scalar_array_set(phi, index, &ref);
  field_scalar_array(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  ref = 1.0/3.0;
  field_scalar_set(phi, index, ref);
  field_scalar(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  /* Halo */
  test_field_halo(phi);

  field_free(phi);
  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test3
 *
 *  Vector order parameter.
 *
 *****************************************************************************/

static int do_test3(void) {

  int nfref = 3;
  int nf;
  int nhalo = 1;
  int index = 1;
  double ref[3] = {1.0, 2.0, 3.0};
  double value[3];
  double array[3];
  field_t * phi = NULL;

  coords_nhalo_set(nhalo);
  coords_init();
  le_init();

  field_create(nfref, "p", &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  field_init(phi, nhalo);

  field_vector_set(phi, index, ref);
  field_vector(phi, index, value);
  assert(fabs(value[0] - ref[0]) < DBL_EPSILON);
  assert(fabs(value[1] - ref[1]) < DBL_EPSILON);
  assert(fabs(value[2] - ref[2]) < DBL_EPSILON);

  field_scalar_array(phi, index, array);
  assert(fabs(array[0] - ref[0]) < DBL_EPSILON);
  assert(fabs(array[1] - ref[1]) < DBL_EPSILON);
  assert(fabs(array[2] - ref[2]) < DBL_EPSILON);

  /* Halo */
  test_field_halo(phi);

  field_free(phi);
  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test5
 *
 *  Tensor order parameter.
 *
 *****************************************************************************/

static int do_test5(void) {

  int nfref = 5;
  int nf;
  int nhalo = 1;
  int index = 1;
  double qref[3][3] = {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, -5.0}};
  double qvalue[3][3];
  double array[NQAB];
  field_t * phi = NULL;

  coords_nhalo_set(nhalo);
  coords_init();
  le_init();

  field_create(nfref, "q", &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  field_init(phi, nhalo);

  field_tensor_set(phi, index, qref);
  field_tensor(phi, index, qvalue);
  assert(fabs(qvalue[X][X] - qref[X][X]) < DBL_EPSILON);
  assert(fabs(qvalue[X][Y] - qref[X][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[X][Z] - qref[X][Z]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][X] - qref[Y][X]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][Y] - qref[Y][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][Z] - qref[Y][Z]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][X] - qref[Z][X]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][Y] - qref[Z][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][Z] - qref[Z][Z]) < DBL_EPSILON);

  /* This is the upper trianle minus the ZZ component */

  field_scalar_array(phi, index, array);
  assert(fabs(array[XX] - qref[X][X]) < DBL_EPSILON);
  assert(fabs(array[XY] - qref[X][Y]) < DBL_EPSILON);
  assert(fabs(array[XZ] - qref[X][Z]) < DBL_EPSILON);
  assert(fabs(array[YY] - qref[Y][Y]) < DBL_EPSILON);
  assert(fabs(array[YZ] - qref[Y][Z]) < DBL_EPSILON);

  /* Halo */
  test_field_halo(phi);

  field_free(phi);
  le_init();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_field_halo
 *
 *****************************************************************************/

static int test_field_halo(field_t * phi) {

  assert(phi);
  
  test_coords_field_set(phi->nf, phi->data, MPI_DOUBLE, test_ref_double1);
  field_memcpy(phi, cudaMemcpyHostToDevice);
 
  field_halo_swap(phi, FIELD_HALO_TARGET);

  field_memcpy(phi, cudaMemcpyDeviceToHost);
  test_coords_field_check(phi->nhcomm, phi->nf, phi->data, MPI_DOUBLE,
			  test_ref_double1);

  return 0;
} 

/*****************************************************************************
 *
 *  do_test_io
 *
 *****************************************************************************/

static int do_test_io(int nf, int io_format) {

  int grid[3] = {1, 1, 1};
  const char * filename = "phi-test-io";

  field_t * phi = NULL;
  io_info_t * iohandler = NULL;

  coords_init();
  le_init();

  if (pe_size() == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  field_create(nf, "phi-test", &phi);
  assert(phi);
  field_init(phi, coords_nhalo());
  field_init_io_info(phi, grid, io_format, io_format); 

  test_coords_field_set(nf, phi->data, MPI_DOUBLE, test_ref_double1);
  field_io_info(phi, &iohandler);
  assert(iohandler);

  io_write_data(iohandler, filename, phi);

  field_free(phi);
  MPI_Barrier(pe_comm());

  field_create(nf, "phi-test", &phi);
  field_init(phi, coords_nhalo());
  field_init_io_info(phi, grid, io_format, io_format);

  field_io_info(phi, &iohandler);
  assert(iohandler);
  io_read_data(iohandler, filename, phi);

  field_halo(phi);
  test_coords_field_check(0, nf, phi->data, MPI_DOUBLE, test_ref_double1);

  MPI_Barrier(pe_comm());
  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "phi-test");

  field_free(phi);
  le_finish();
  coords_finish();

  return 0;
}
