/*****************************************************************************
 *
 *  test_hydro.c
 *
 *  Unit test for hydrodynamics object. Tests for the Lees Edwards
 *  transformations are sadly lacking.
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"
#include "hydro_s.h"

typedef int (* halo_test_ft) (int, int, int, int, double *);

static int do_test1(void);
static int do_test_halo1(int nhalo, int nhcomm);
static int do_test_io1(void);
static int testf1(int ic, int jc, int kc, int n, double * ref);

int test_field_set(int nf, double * f, halo_test_ft fset);
int test_field_check(int nf, int nhcomm, double * f, halo_test_ft fref);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  do_test1();
  do_test_halo1(1, 1);
  do_test_halo1(2, 2);
  do_test_halo1(2, 1);
  do_test_io1();

  pe_finalise();
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *****************************************************************************/

static int do_test1(void) {

  hydro_t * hydro = NULL;

  int index;
  const double force[3] = {1.0, 2.0, 3.0};
  const double u[3] = {-1.0, -2.0, -3.0};
  double check[3] = {0.0, 0.0, 0.0};

  coords_init();
  le_init();

  hydro_create(1, &hydro);
  assert(hydro);

  index = coords_index(1, 1, 1);
  hydro_f_local_set(hydro, index, force);
  hydro_f_local(hydro, index, check);
  assert(fabs(force[X] - check[X]) < DBL_EPSILON);
  assert(fabs(force[Y] - check[Y]) < DBL_EPSILON);
  assert(fabs(force[Z] - check[Z]) < DBL_EPSILON);

  hydro_u_set(hydro, index, u);
  hydro_u(hydro, index, check);
  assert(fabs(u[X] - check[X]) < DBL_EPSILON);
  assert(fabs(u[Y] - check[Y]) < DBL_EPSILON);
  assert(fabs(u[Z] - check[Z]) < DBL_EPSILON);

  hydro_free(hydro);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo1
 *
 *****************************************************************************/

static int do_test_halo1(int nhalo, int nhcomm) {

  hydro_t * hydro = NULL;

  coords_nhalo_set(nhalo);
  coords_init();
  le_init();

  hydro_create(nhcomm, &hydro);
  assert(hydro);

  test_field_set(hydro->nf, hydro->u, testf1);
  hydro_u_halo(hydro);
  test_field_check(hydro->nf, nhcomm, hydro->u, testf1);

  hydro_free(hydro);

  le_finish();
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io1
 *
 *****************************************************************************/

static int do_test_io1(void) {

  int grid[3] = {1, 1, 1};
  char * filename = "hydro-test-io";
  struct io_info_t * iohandler = NULL;

  coords_init();
  le_init();

  if (pe_size() == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  hydro_create(1, &hydro_);
  assert(hydro_);

  hydro_init_io_info(hydro_, grid, IO_FORMAT_DEFAULT, IO_FORMAT_DEFAULT);
  test_field_set(hydro_->nf, hydro_->u, testf1);

  hydro_io_info(hydro_, &iohandler);
  assert(iohandler);
  io_write(filename, iohandler);

  io_remove(filename, iohandler);

  hydro_free(hydro_);
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
