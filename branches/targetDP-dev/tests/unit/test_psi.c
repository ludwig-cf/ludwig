/*****************************************************************************
 *
 *  test_psi.c
 *
 *  Unit test for electrokinetic quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2014 The University of Edinburgh
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

#include "test_coords_field.h"
#include "tests.h"

static int testf2(int ic, int jc, int kc, int n, void * ref);
static int do_test1(void);
static int do_test2(void);
static int do_test_halo1(void);
static int do_test_halo2(void);
static int do_test_bjerrum(void);
static int do_test_ionic_strength(void);
static int do_test_io1(void);

/*****************************************************************************
 *
 *  test_psi_suite
 *
 *****************************************************************************/

int test_psi_suite(void) {

  pe_init_quiet();

  do_test1();
  do_test2();
  do_test_halo1();
  do_test_halo2();
  do_test_io1();
  do_test_bjerrum();
  do_test_ionic_strength();

  info("PASS     ./unit/test_psi\n");
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *  Test object creation/deletion, and the various access functions.
 *
 *****************************************************************************/

static int do_test1(void) {

  int nk;
  int iv, n;
  double e, diff;
  double valency[3] = {1, 2, 3};
  double diffusivity[3] = {1.0, 2.0, 3.0};
  double eunit = -1.0;

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

  psi_unit_charge(psi, &e);
  assert(fabs(e - 1.0) < DBL_EPSILON); /* Default unit = 1.0 */
  psi_unit_charge_set(psi, eunit);
  psi_unit_charge(psi, &e);
  assert(fabs(eunit - e) < DBL_EPSILON);

  psi_free(psi);

  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test2
 *
 *  Check access to the lattice-based quantities.
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
 *  Take the default system size with nhalo = 2 and nk = 3, and
 *  check the halo swap.
 *
 *****************************************************************************/

static int do_test_halo1(void) {

  int nk;
  int nhalo = 2;
  psi_t * psi;

  coords_nhalo_set(nhalo);
  coords_init();

  nk = 3;
  psi_create(nk, &psi);
  assert(psi);

  test_coords_field_set(1, psi->psi, MPI_DOUBLE, test_ref_double1);
  psi_halo_psi(psi);
  test_coords_field_check(nhalo, 1, psi->psi, MPI_DOUBLE, test_ref_double1);

  test_coords_field_set(nk, psi->rho, MPI_DOUBLE, test_ref_double1);
  psi_halo_rho(psi);
  test_coords_field_check(nhalo, nk, psi->rho, MPI_DOUBLE, test_ref_double1);

  test_coords_field_set(nk, psi->rho, MPI_DOUBLE, testf2);
  psi_halo_rho(psi);
  test_coords_field_check(nhalo, nk, psi->rho, MPI_DOUBLE, testf2);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo2
 *
 *  Check the halo swap in a one-dimensional decomposition.
 *
 *****************************************************************************/

static int do_test_halo2(void) {

  int nk;
  int grid[3];
  int nhalo = 3;
  psi_t * psi;

  /* Use a 1-d decomposition, which increases the number of
   * MPI tasks in one direction cf. the default. */

  grid[0] = 1;
  grid[1] = pe_size();
  grid[2] = 1;

  coords_decomposition_set(grid);
  coords_nhalo_set(nhalo);
  coords_init();

  nk = 2;
  psi_create(nk, &psi);
  assert(psi);

  test_coords_field_set(1, psi->psi, MPI_DOUBLE, test_ref_double1);
  psi_halo_psi(psi);
  test_coords_field_check(nhalo, 1, psi->psi, MPI_DOUBLE, test_ref_double1);

  test_coords_field_set(nk, psi->rho, MPI_DOUBLE, test_ref_double1);
  psi_halo_rho(psi);
  test_coords_field_check(nhalo, nk, psi->rho, MPI_DOUBLE, test_ref_double1);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io1
 *
 *  Note that the io functions must use the psi_ object at the moment.
 *  Take default (i.e., binary) write, and specify explicitly binary
 *  read.
 * 
 *****************************************************************************/

static int do_test_io1(void) {

  int nk;
  int grid[3] = {1, 1, 1};
  char * filename = "psi-test-io";

  psi_t * psi = NULL;
  io_info_t * iohandler = NULL;

  coords_init();

  if (pe_size() == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  nk = 2;
  psi_create(nk, &psi);
  assert(psi);
  psi_init_io_info(psi, grid, IO_FORMAT_DEFAULT, IO_FORMAT_DEFAULT);

  test_coords_field_set(1, psi->psi, MPI_DOUBLE, test_ref_double1);
  test_coords_field_set(nk, psi->rho, MPI_DOUBLE, test_ref_double1);

  psi_io_info(psi, &iohandler);
  assert(iohandler);
  io_write_data(iohandler, filename,  psi);

  psi_free(psi);
  MPI_Barrier(pe_comm());

  /* Recreate, and read. This zeros out all the fields, so they
   * must be read correctly to pass. */

  psi_create(nk, &psi);
  psi_init_io_info(psi, grid, IO_FORMAT_BINARY, IO_FORMAT_BINARY);

  psi_io_info(psi, &iohandler);
  assert(iohandler);
  io_read_data(iohandler, filename, psi);

  psi_halo_psi(psi);
  psi_halo_rho(psi);

  /* Zero halo region required */
  test_coords_field_check(0, 1, psi->psi, MPI_DOUBLE, test_ref_double1);
  test_coords_field_check(0, nk, psi->rho, MPI_DOUBLE, test_ref_double1);

  MPI_Barrier(pe_comm());
  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "psi");

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_bjerrum
 *
 *  Test the bjerrum length comes out right.
 *  l_B = e^2 / 4\pi epsilon KT
 *
 *  We set out unit charge to be 1 in lattice units, and a plausable
 *  lattice Boltzmann temperature of 10^-05; the units of permeativity
 *  are still somewhat open to investigation...
 *
 *  At the moment I have the famous 41.4 which is the dielectric
 *  *isotropy* used for blue phases scaled by an arbitrary 1000.
 *
 *  Also test the Debye length for unit ionic strength while we
 *  are at it.
 *
 *****************************************************************************/

static int do_test_bjerrum(void) {

  psi_t * psi = NULL;
  double eref = 1.0;
  double epsilonref = 41.4*1000.0;
  double ktref = 0.00001;
  double tmp, lbref, ldebyeref;

  coords_init();
  psi_create(2, &psi);

  psi_beta_set(psi, 1.0/ktref);
  psi_beta(psi, &tmp);
  assert(fabs(1.0/ktref - tmp) < DBL_EPSILON);

  psi_epsilon_set(psi, epsilonref);
  psi_epsilon(psi, &tmp);
  assert(fabs(tmp - epsilonref) < DBL_EPSILON);

  psi_unit_charge_set(psi, eref);
  psi_unit_charge(psi, &tmp);
  assert(fabs(tmp - eref) < DBL_EPSILON);

  lbref = eref*eref / (4.0*M_PI*epsilonref*ktref);
  psi_bjerrum_length(psi, &tmp);
  assert(fabs(lbref - tmp) < DBL_EPSILON);

  /* For unit ionic strength */
  ldebyeref = 1.0 / sqrt(8.0*M_PI*lbref);
  psi_debye_length(psi, 1.0, &tmp);
  assert(fabs(ldebyeref - tmp) < DBL_EPSILON);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_ionic_strength
 *
 *  Test the calculation of the ionic strength. We require just
 *  valency and charge density for a given number of species. 
 *
 *****************************************************************************/

static int do_test_ionic_strength(void) {

  int n, nk = 2;
  psi_t * psi = NULL;

  int index = 1;             /* Lattice point will be present */
  int valency[2] = {+2, -2};
  double rho[2] = {1.0, 2.0};
  double rhoi;
  double expect;

  coords_init();
  psi_create(nk, &psi);

  for (n = 0; n < nk; n++) {
    psi_valency_set(psi, n, valency[n]);
    psi_rho_set(psi, index, n, rho[n]);
  }

  expect = 0.5*(pow(valency[0], 2)*rho[0] + pow(valency[1], 2)*rho[1]);

  psi_ionic_strength(psi, index, &rhoi);
  assert(fabs(expect - rhoi) < DBL_EPSILON);

  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  testf2
 *
 *  A 'wall' function perioidic in z-direction.
 *
 *****************************************************************************/

static int testf2(int ic, int jc, int kc, int n, void * buf) {

  double * ref = buf;

  assert(ref);

  *ref = -1.0;

  if (kc == 1 || kc == 0) *ref = 1.0;
  if (kc == N_total(Z) || kc == N_total(Z) + 1) *ref = 1.0;

  return 0;
}
