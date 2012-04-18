/*****************************************************************************
 *
 *  test_psi.c
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "psi.h"
#include "psi_s.h"

typedef int (* halo_test_ft) (int, int, int, int, double *);

static int testf1(int ic, int jc, int kc, int n, double * ref);
static int do_test1(void);
static int do_test2(void);
static int do_test_halo1(void);
static int do_test_halo2(void);
static int do_test_io1(void);

int test_field_set(int nf, double * f, halo_test_ft fset);
int test_field_check(int nf, double * f, halo_test_ft fref);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  do_test1();
  do_test2();
  do_test_halo1();
  do_test_halo2();
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

  int nk;
  int iv, n;
  double diff;
  double valency[3] = {1, 2, 3};
  double diffusivity[3] = {1.0, 2.0, 3.0};

  psi_t * psi;

  coords_init();

  nk = 3;
  psi_create(nk, &psi);
  assert(psi);
  psi_nk(psi, &n);
  assert(n == 3);

  for (n = 0; n < nk; n++) {
    psi_valency_set(psi, n, valency[n]);
    psi_valency(psi, n, &iv);
    assert(iv == valency[n]);
    psi_diffusivity_set(psi, n, diffusivity[n]);
    psi_diffusivity(psi, n, &diff);
    assert(fabs(diff - diffusivity[n]) < DBL_EPSILON);
  }

  psi_free(psi);

  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test2
 *
 *****************************************************************************/

static int do_test2(void) {

  int nk = 2;
  int iv, n;
  int index;
  double diff;
  double ref, value;
  double valency[2] = {1, 2};
  double diffusivity[2] = {1.0, 2.0};

  psi_t * psi;

  coords_init();

  psi_create(nk, &psi);
  assert(psi);
  psi_nk(psi, &n);
  assert(n == 2);

  for (n = 0; n < nk; n++) {
    psi_valency_set(psi, n, valency[n]);
    psi_valency(psi, n, &iv);
    assert(iv == valency[n]);
    psi_diffusivity_set(psi, n, diffusivity[n]);
    psi_diffusivity(psi, n, &diff);
    assert(fabs(diff - diffusivity[n]) < DBL_EPSILON);
  }

  index = 1;
  ref = 1.0;
  psi_psi_set(psi, index, ref);
  psi_psi(psi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  for (n = 0; n < nk; n++) {
    ref = 1.0 + n;
    psi_rho_set(psi, index, n, ref);
    psi_rho(psi, index, n, &value);
    assert(fabs(value - ref) < DBL_EPSILON);
  }

  ref = 1.0 + 4.0;
  psi_rho_elec(psi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo1
 *
 *****************************************************************************/

static int do_test_halo1(void) {

  int nk;
  psi_t * psi;

  coords_nhalo_set(2);
  coords_init();

  nk = 3;
  psi_create(nk, &psi);
  assert(psi);

  test_field_set(1, psi->psi, testf1);
  psi_halo(1, psi->psi, psi->psihalo);
  test_field_check(1, psi->psi, testf1);

  test_field_set(nk, psi->rho, testf1);
  psi_halo(nk, psi->rho, psi->rhohalo);
  test_field_check(nk, psi->rho, testf1);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo2
 *
 *****************************************************************************/

static int do_test_halo2(void) {

  int nk;
  int grid[3];
  psi_t * psi;

  /* Use a 1-d decomposition, which increases the number of
   * MPI tasks in one direction cf. the default. */

  grid[0] = 1;
  grid[1] = pe_size();
  grid[2] = 1;

  coords_decomposition_set(grid);
  coords_nhalo_set(3);
  coords_init();

  nk = 2;
  psi_create(nk, &psi);
  assert(psi);

  test_field_set(1, psi->psi, testf1);
  psi_halo_psi(psi);
  test_field_check(1, psi->psi, testf1);

  test_field_set(nk, psi->rho, testf1);
  psi_halo_rho(psi);
  test_field_check(nk, psi->rho, testf1);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io1
 *
 *  Note that the io functions must use the psi_ object.
 * 
 *****************************************************************************/

static int do_test_io1(void) {

  int nk;
  int grid[3] = {1, 1, 1};
  char * filename = "psi-test-io";

  coords_init();

  nk = 2;
  psi_create(nk, &psi_);
  psi_init_io_info(psi_, grid);

  test_field_set(1, psi_->psi, testf1);
  test_field_set(nk, psi_->rho, testf1);
  io_write(filename, psi_->info);

  psi_free(psi_);
  MPI_Barrier(pe_comm());

  /* Recreate, and read */

  psi_create(nk, &psi_);
  psi_init_io_info(psi_, grid);
  io_read(filename, psi_->info);

  psi_halo_psi(psi_);
  psi_halo_rho(psi_);

  test_field_check(1, psi_->psi, testf1);
  test_field_check(nk, psi_->rho, testf1);
  io_remove(filename, psi_->info);

  psi_free(psi_);
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

int test_field_check(int nf, double * f, halo_test_ft fref) {

  int n;
  int nhalo;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index;

  double ref;               /* Reference function value */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  /* Check all points, i.e., interior points should not have changed,
   * and halo point should be correctly updated. Some of the differences
   * coming from periodic boundaries are slightly largely than DBL_EPSILON,
   * but FLT_EPSILON should catch gross errors. */

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nf; n++) {
	  fref(noffst[X]+ic, noffst[Y]+jc, noffst[Z]+kc, n, &ref);
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
  
  *ref = cos(2.0*pi_*ic/L(X)) + cos(2.0*pi_*jc/L(Y)) + cos(2.0*pi_*kc/L(Z));
  *ref += 1.0*n;

  return 0;
}
