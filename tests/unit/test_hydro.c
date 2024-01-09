/*****************************************************************************
 *
 *  test_hydro.c
 *
 *  Unit test for hydrodynamics object. Tests for the Lees Edwards
 *  transformations are sadly lacking.
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2022 The University of Edinburgh
 *
 *  Contributing authors
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "hydro.h"

#include "test_coords_field.h"
#include "tests.h"

static int do_test1(pe_t * pe);
static int do_test_halo1(pe_t * pe, int nhalo, int nhcomm,
			 field_halo_enum_t haloscheme);
static int do_test_io1(pe_t * pe, int io_format);

int test_hydro_rho(pe_t * pe);

/*****************************************************************************
 *
 *  test_hydro_suite
 *
 *****************************************************************************/

int test_hydro_suite(void) {

  int ndevice = 0;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  tdpGetDeviceCount(&ndevice);

  do_test1(pe);
  do_test_halo1(pe, 1, 1, FIELD_HALO_TARGET);
  do_test_halo1(pe, 2, 2, FIELD_HALO_TARGET);
  do_test_halo1(pe, 2, 1, FIELD_HALO_TARGET);

  if (ndevice == 0) {
    do_test_halo1(pe, 1, 1, FIELD_HALO_OPENMP);
    do_test_halo1(pe, 2, 1, FIELD_HALO_OPENMP);
  }

  test_hydro_rho(pe);

  do_test_io1(pe, IO_FORMAT_DEFAULT);
  do_test_io1(pe, IO_FORMAT_ASCII);
  do_test_io1(pe, IO_FORMAT_BINARY);

  pe_info(pe, "PASS     ./unit/test_hydro\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *****************************************************************************/

static int do_test1(pe_t * pe) {

  int index;
  const double force[3] = {1.0, 2.0, 3.0};
  const double u[3] = {-1.0, -2.0, -3.0};
  double check[3] = {0.0, 0.0, 0.0};

  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  hydro_options_t opts = hydro_options_default();
  hydro_t * hydro = NULL;

  assert(pe);
  assert(NHDIM == 3);

  cs_create(pe, &cs);
  cs_init(cs);
  lees_edw_create(pe, cs, NULL, &le);

  hydro_create(pe, cs, le, &opts, &hydro);
  assert(hydro);
  assert(hydro->nhcomm == 1);

  index = cs_index(cs, 1, 1, 1);
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

  /* Smoke tests */
  hydro_u_zero(hydro, u);
  hydro_f_zero(hydro, force);

  hydro_free(hydro);

  lees_edw_free(le);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo1
 *
 *****************************************************************************/

static int do_test_halo1(pe_t * pe, int nhalo, int nhcomm,
			 field_halo_enum_t haloscheme) {

  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  hydro_options_t opts = hydro_options_default();
  hydro_t * hydro = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  lees_edw_create(pe, cs, NULL, &le);

  opts.nhcomm = nhcomm;
  opts.u.nhcomm = nhcomm;
  opts.u.haloscheme = haloscheme;

  hydro_create(pe, cs, le, &opts, &hydro);
  assert(hydro);

  test_coords_field_set(cs, NHDIM, hydro->u->data, MPI_DOUBLE, test_ref_double1);
  hydro_memcpy(hydro, tdpMemcpyHostToDevice);

  hydro_u_halo(hydro);

  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);
  test_coords_field_check(cs, nhcomm, NHDIM, hydro->u->data, MPI_DOUBLE,
			  test_ref_double1);

  hydro_free(hydro);

  lees_edw_free(le);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io1
 *
 *****************************************************************************/

static int do_test_io1(pe_t * pe, int io_format) {

  int grid[3] = {1, 1, 1};
  const char * filename = "hydro-test-io";

  MPI_Comm comm;

  cs_t * cs = NULL;
  io_info_t * iohandler = NULL;

  hydro_options_t opts = hydro_options_default();
  hydro_t * hydro = NULL;

  assert(pe);

  pe_mpi_comm(pe, &comm);

  cs_create(pe, &cs);
  cs_init(cs);

  if (pe_mpi_size(pe) == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  hydro_create(pe, cs, NULL, &opts, &hydro);
  assert(hydro);

  hydro_init_io_info(hydro, grid, io_format, io_format);
  test_coords_field_set(cs, NHDIM, hydro->u->data, MPI_DOUBLE, test_ref_double1);

  hydro_io_info(hydro, &iohandler);
  assert(iohandler);

  io_write_data(iohandler, filename, hydro);

  MPI_Barrier(comm);
  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "vel");

  hydro_free(hydro);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_hydro_rho
 *
 *****************************************************************************/

int test_hydro_rho(pe_t * pe) {

  cs_t * cs = NULL;

  hydro_options_t opts = hydro_options_default();
  hydro_t * hydro = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);

  hydro_create(pe, cs, NULL, &opts, &hydro);

  assert(hydro->rho);

  {
    int index   = hydro->nsite - 1;
    double rho0 = 2.0;
    double rho  = 0.0;

    hydro_rho_set(hydro, index, rho0);
    hydro_rho(hydro, index, &rho);
    assert(fabs(rho - rho0) < DBL_EPSILON);
  }

  hydro_free(hydro);
  cs_free(cs);

  return 0;
}
