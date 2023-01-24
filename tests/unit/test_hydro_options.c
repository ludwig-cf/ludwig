/*****************************************************************************
 *
 *  test_hydro_options.c
 *
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "hydro_options.h"

int test_hydro_options_default(void);
int test_hydro_options_nhalo(void);
int test_hydro_options_haloscheme(void);

/*****************************************************************************
 *
 *  test_hydro_options_suite
 *
 *****************************************************************************/

int test_hydro_options_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_hydro_options_default();
  test_hydro_options_nhalo();
  test_hydro_options_haloscheme();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_hydro_options_default
 *
 *****************************************************************************/

int test_hydro_options_default(void) {

  int ifail = 0;
  hydro_options_t opts = hydro_options_default();

  assert(opts.nhcomm == 1);

  assert(opts.rho.ndata == 1);
  assert(opts.rho.nhcomm == 1);
  assert(opts.rho.haloscheme == FIELD_HALO_TARGET);
  ifail = field_options_valid(&opts.rho);
  assert(ifail == 1);

  assert(opts.u.ndata == 3);
  assert(opts.u.nhcomm == 1);
  assert(opts.u.haloscheme == FIELD_HALO_TARGET);
  ifail = field_options_valid(&opts.u);
  assert(ifail == 1);

  assert(opts.force.ndata == 3);
  assert(opts.force.nhcomm == 1);
  assert(opts.force.haloscheme == FIELD_HALO_TARGET);
  ifail = field_options_valid(&opts.force);
  assert(ifail == 1);

  assert(opts.eta.ndata == 1);
  assert(opts.eta.nhcomm == 1);
  assert(opts.eta.haloscheme == FIELD_HALO_TARGET);
  ifail = field_options_valid(&opts.eta);
  assert(ifail == 1);

  return ifail;
}

/*****************************************************************************
 *
 *  test_hydro_options_nhalo
 *
 *****************************************************************************/

int test_hydro_options_nhalo(void) {

  int ifail = 0;

  {
    int nhalo = 2;
    hydro_options_t opts = hydro_options_nhalo(nhalo);
    assert(opts.rho.nhcomm   == nhalo);
    assert(opts.u.nhcomm     == nhalo);
    assert(opts.force.nhcomm == nhalo);
    assert(opts.eta.nhcomm   == nhalo);
    if (opts.rho.nhcomm != nhalo) ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_hydro_options_haloscheme
 *
 *****************************************************************************/

int test_hydro_options_haloscheme(void) {

  int ifail = 0;

  {
    field_halo_enum_t hs = FIELD_HALO_OPENMP;
    hydro_options_t opts = hydro_options_haloscheme(hs);

    assert(opts.rho.haloscheme   == hs);
    assert(opts.u.haloscheme     == hs);
    assert(opts.force.haloscheme == hs);
    assert(opts.eta.haloscheme   == hs);
    if (opts.rho.haloscheme != hs) ifail = -1;
  }

  return ifail;
}
