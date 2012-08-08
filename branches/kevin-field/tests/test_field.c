/*****************************************************************************
 *
 *  test_field.c
 *
 *  Unit test for field structure.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
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


static int do_test1(void);
static int do_test3(void);
static int do_test5(void);
static int do_test_io(int nf, int io_format);
static int test_field_halo(field_t * phi);
static int testf1(int ic, int jc, int kc, int n, double * ref);

typedef int (* halo_test_ft) (int, int, int, int, double *);

int test_field_set(int nf, double * f, halo_test_ft fset);
int test_field_check(int nf, int nhcomm, double * f, halo_test_ft fref);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main (int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  info("\nOrder parameter tests...\n");

  do_test1();
  do_test3();
  do_test5();

  do_test_io(1, IO_FORMAT_ASCII);
  do_test_io(1, IO_FORMAT_BINARY);
  do_test_io(5, IO_FORMAT_ASCII);
  do_test_io(5, IO_FORMAT_BINARY);

  pe_finalise();
  MPI_Finalize();

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

  test_field_set(phi->nf, phi->data, testf1);
  field_halo(phi);
  test_field_check(phi->nf, phi->nhcomm, phi->data, testf1);

  return 0;
} 

/*****************************************************************************
 *
 *  do_test_io
 *
 *****************************************************************************/

static int do_test_io(int nf, int io_format) {

  int grid[3] = {1, 1, 1};
  char * filename = "phi-test-io";

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

  test_field_set(nf, phi->data, testf1);
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
  test_field_check(nf, phi->nhcomm, phi->data, testf1);
  io_remove(filename, iohandler);

  field_free(phi);
  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_field_set
 *
 *****************************************************************************/

int test_field_set(int nf, double * f, halo_test_ft fset) {

  int n;
  int nhalo;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  /* Set values in the domain proper (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        for (n = 0; n < nf; n++) {
          fset(noffst[X]+ic, noffst[Y]+jc, noffst[Z]+kc, n, f + nf*index + n);
        }

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_field_check
 *
 *****************************************************************************/

int test_field_check(int nf, int nhcomm, double * f, halo_test_ft fref) {

  int n;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index;

  double ref;               /* Reference function value */

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  /* Check all points, i.e., interior points should not have changed,
   * and halo point should be correctly updated. Some of the differences
   * coming from periodic boundaries are slightly largely than DBL_EPSILON,
   * but FLT_EPSILON should catch gross errors. */

  for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
    for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
      for (kc = 1 - nhcomm; kc <= nlocal[Z] + nhcomm; kc++) {

        index = coords_index(ic, jc, kc);

        for (n = 0; n < nf; n++) {
          fref(noffst[X]+ic, noffst[Y]+jc, noffst[Z]+kc, n, &ref);
          if (fabs(f[nf*index + n] - ref) > FLT_EPSILON) {
            verbose("%2d %2d %2d %2d %f %f\n", ic, jc, kc, n, ref, f[nf*index + n]);
          }
          assert(fabs(f[nf*index + n] - ref) < FLT_EPSILON);
        }

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  testf1
 *
 *****************************************************************************/

static int testf1(int ic, int jc, int kc, int n, double * ref) {

  assert(ref);
  
  *ref = cos(2.0*M_PI*ic/L(X)) + cos(2.0*M_PI*jc/L(Y)) + cos(2.0*M_PI*kc/L(Z));
  *ref += 1.0*n;

  return 0;
}
