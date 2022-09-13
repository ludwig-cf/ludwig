/*****************************************************************************
 *
 *  test_phi_ch.c
 *
 *  That is, unit test of phi_cahn_hilliard.c
 *  A free energy is required so use fe_null.h.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "fe_null.h"
#include "physics.h"
#include "phi_cahn_hilliard.h"

int test_phi_ch_create(pe_t * pe);
int test_phi_cahn_hilliard(pe_t * pe);


/*****************************************************************************
 *
 *  test_ph_ch_suite
 *
 *****************************************************************************/

int test_phi_ch_suite(void) {

  pe_t * pe = NULL;
  physics_t * phys = NULL;  /* Dependency via Lees Edwards */

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  physics_create(pe, &phys);

  test_phi_ch_create(pe);
  test_phi_cahn_hilliard(pe);

  pe_info(pe, "PASS     ./unit/test_phi_ch\n");

  physics_free(phys);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_ch_create
 *
 *****************************************************************************/

int test_phi_ch_create(pe_t * pe) {

  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  {
    lees_edw_info_t opts = {0};
    lees_edw_create(pe, cs, &opts, &le);
  }

  {
    phi_ch_info_t info = {.conserve = 0};
    phi_ch_t * ch = NULL;

    phi_ch_create(pe, cs, le, &info, &ch);
    assert(ch);

    assert(ch->info.conserve == info.conserve);
    assert(ch->pe == pe);
    assert(ch->cs == cs);
    assert(ch->csum == NULL);
    assert(ch->le == le);
    assert(ch->flux);

    phi_ch_free(ch);
  }

  {
    phi_ch_info_t info = {.conserve = 1};
    phi_ch_t * ch = NULL;

    phi_ch_create(pe, cs, le, &info, &ch);
    assert(ch);

    assert(ch->info.conserve == info.conserve);
    assert(ch->pe == pe);
    assert(ch->cs == cs);
    assert(ch->csum);
    assert(ch->le == le);
    assert(ch->flux);

    phi_ch_free(ch);
  }

  lees_edw_free(le);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_cahn_hilliard
 *
 *  Minimum requirement of an order parameter and free energy.
 *
 *****************************************************************************/

int test_phi_cahn_hilliard(pe_t * pe) {

  cs_t * cs = NULL;
  lees_edw_t * le = NULL;
  field_t * field = NULL;
  fe_null_t * fe = NULL;

  cs_create(pe, &cs);
  cs_init(cs);

  fe_null_create(pe, &fe);

  {
    lees_edw_info_t opts = {0};
    lees_edw_create(pe, cs, &opts, &le);
  }

  {
    /* Order parameter field */
    field_options_t opts = field_options_default();
    field_create(pe, cs, le, "phi", &opts, &field);
  }

  {
    /* No test as such. */
    phi_ch_info_t info = {.conserve = 0};
    phi_ch_t * ch = NULL;

    phi_ch_create(pe, cs, le, &info, &ch);

    phi_cahn_hilliard(ch, (fe_t *) fe, field, NULL, NULL, NULL);

    phi_ch_free(ch);
  }

  field_free(field);
  lees_edw_free(le);
  fe_null_free(fe);
  cs_free(cs);

  return 0;
}
