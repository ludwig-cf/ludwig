/*****************************************************************************
 *
 *  test_psi.c
 *
 *  Unit test for electrokinetic quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "psi.h"

int test_psi_initialise(pe_t * pe);
int test_psi_create(pe_t * pe);
int test_psi_psi_set(pe_t * pe);
int test_psi_rho_set(pe_t * pe);
int test_psi_halo_psi(pe_t * pe);
int test_psi_halo_rho(pe_t * pe);
int test_psi_ionic_strength(pe_t * pe);

/*****************************************************************************
 *
 *  test_psi_suite
 *
 *****************************************************************************/

int test_psi_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_psi_initialise(pe);
  test_psi_create(pe);
  test_psi_psi_set(pe);
  test_psi_rho_set(pe);
  test_psi_halo_psi(pe);
  test_psi_halo_rho(pe);
  test_psi_ionic_strength(pe);

  pe_info(pe, "PASS     ./unit/test_psi\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_initialise
 *
 *****************************************************************************/

int test_psi_initialise(pe_t * pe) {

  int nhalo = 1;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  {
    /* With some non-default values ... */
    int nsites = 0;
    psi_options_t opts = psi_options_default(nhalo);
    psi_t psi = {0};

    opts.e        = 2.0;
    opts.beta     = 0.5;
    opts.epsilon1 = 0.2;
    opts.epsilon2 = 0.3;
    opts.e0[X]    = 4.0;
    opts.e0[Y]    = 5.0;
    opts.e0[Z]    = 6.0;

    psi_initialise(pe, cs, &opts, &psi);

    cs_nsites(cs, &nsites);

    /* Check existing structure */
    assert(psi.pe     == pe);
    assert(psi.cs     == cs);
    assert(psi.nk     == opts.nk);
    assert(psi.nsites == nsites);
    assert(psi.psi    != NULL);
    assert(psi.rho    != NULL);

    assert(fabs(psi.diffusivity[0] - opts.diffusivity[0]) < DBL_EPSILON);
    assert(fabs(psi.diffusivity[1] - opts.diffusivity[1]) < DBL_EPSILON);
    assert(psi.valency[0] == opts.valency[0]);
    assert(psi.valency[1] == opts.valency[1]);

    /* Physical quantities */
    assert(fabs(psi.e - opts.e) < DBL_EPSILON);
    assert(fabs(psi.epsilon  - opts.epsilon1) < DBL_EPSILON);
    assert(fabs(psi.epsilon2 - opts.epsilon2) < DBL_EPSILON);
    assert(fabs(psi.beta - opts.beta) < DBL_EPSILON);
    assert(fabs(psi.e0[X] - opts.e0[X]) < DBL_EPSILON);
    assert(fabs(psi.e0[Y] - opts.e0[Y]) < DBL_EPSILON);
    assert(fabs(psi.e0[Z] - opts.e0[Z]) < DBL_EPSILON);

    /* Solver options */
    /* Assume correctly covered in solver options tests ... */
    assert(psi.solver.psolver == PSI_POISSON_SOLVER_SOR);

    /* Nernst Planck */
    assert(psi.multisteps == opts.nsmallstep);
    assert(fabs(psi.diffacc - opts.diffacc) < DBL_EPSILON);

    /* Other */
    assert(psi.method == opts.method);

    psi_finalise(&psi);
  }

  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_create
 *
 *****************************************************************************/

int test_psi_create(pe_t * pe) {

  int nhalo = 2;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  {
    psi_options_t opts = psi_options_default(nhalo);
    psi_t * psi = NULL;

    psi_create(pe, cs, &opts, &psi);
    assert(psi);
    assert(psi->psi);
    assert(psi->rho);

    psi_free(&psi);
    assert(psi == NULL);
  }

  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_psi_set
 *
 *****************************************************************************/

int test_psi_psi_set(pe_t * pe) {

  int nhalo = 2;
  int ntotal[3] = {64, 4, 4};
  psi_options_t opts = psi_options_default(nhalo);
  psi_t * psi = NULL;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  psi_create(pe, cs, &opts, &psi);
  assert(psi->psi->data);

  {
    /* potential */
    int index = 1;
    double psi0 = 2.0;
    double value = 0.0;
    psi_psi_set(psi, index, psi0);
    psi_psi(psi, index, &value);
    assert(fabs(value - psi0) < DBL_EPSILON);
  }

  psi_free(&psi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  psi_rho_set
 *
 *****************************************************************************/

int test_psi_rho_set(pe_t * pe) {

  int nhalo = 2;
  int ntotal[3] = {64, 4, 4};
  psi_options_t opts = psi_options_default(nhalo);
  psi_t * psi = NULL;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  psi_create(pe, cs, &opts, &psi);
  assert(psi->rho->data);

  /* Charge densities */
  for (int n = 0; n < psi->nk; n++) {
    int index = 1 + n;
    double rho0 = 6.0 + 1.0*n;
    double value = 0.0;
    psi_rho_set(psi, index, n, rho0);
    psi_rho(psi, index, n, &value);
    assert(fabs(value - rho0) < DBL_EPSILON);
  }

  psi_free(&psi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_halo_psi
 *
 *****************************************************************************/

int test_psi_halo_psi(pe_t * pe) {

  int ifail = 0;
  int nhalo = 2;
  int ntotal[3] = {16, 16, 16};
  psi_options_t opts = psi_options_default(nhalo);
  psi_t * psi = NULL;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  psi_create(pe, cs, &opts, &psi);

  /* Provide uniform values ... */
  {
    int nlocal[3] = {0};
    double psi0 = 2.0;
    cs_nlocal(cs, nlocal);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  psi_psi_set(psi, index, psi0);
	}
      }
    }
  }

  psi_halo_psi(psi);

  /* Check ... */
  {
    int nlocal[3] = {0};
    double psi0 = 2.0;
    cs_nlocal(cs, nlocal);

    for (int ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
      for (int jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (int kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double value = 0.0;
	  psi_psi(psi, index, &value);
	  assert(fabs(value - psi0) < DBL_EPSILON);
	  if (fabs(value - psi0) > DBL_EPSILON) ifail += 1;
	}
      }
    }
  }

  psi_free(&psi);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_halo_rho
 *
 *****************************************************************************/

int test_psi_halo_rho(pe_t * pe) {

  int ifail = 0;
  int nhalo = 1;
  int ntotal[3] = {16, 16, 16};
  psi_options_t opts = psi_options_default(nhalo);
  psi_t * psi = NULL;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  psi_create(pe, cs, &opts, &psi);

  /* Provide uniform values per chareged species ... */
  {
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  for (int n = 0; n < psi->nk; n++) {
	    double rho0 = 6.0 + 1.0*n;
	    psi_rho_set(psi, index, n, rho0);
	  }
	}
      }
    }
  }

  psi_halo_rho(psi);

  /* Check ... */
  {
    int nlocal[3] = {0};
    cs_nlocal(cs, nlocal);

    for (int ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
      for (int jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (int kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  for (int n = 0; n < psi->nk; n++) {
	    double rho0 = 6.0 + 1.0*n;
	    double value = 0.0;
	    psi_rho(psi, index, n, &value);
	    assert(fabs(value - rho0) < DBL_EPSILON);
	    if (fabs(value - rho0) > DBL_EPSILON) ifail += 1;
	  }
	}
      }
    }
  }

  psi_free(&psi);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_ionic_strength
 *
 *****************************************************************************/

int test_psi_ionic_strength(pe_t * pe) {
  
  int nhalo = 1;
  int ntotal[3] = {8, 8, 8};
  psi_options_t opts = psi_options_default(nhalo);
  psi_t * psi = NULL;
  cs_t * cs = NULL;

  /* We require a valency and a charge density ... */ 
  int valency[2] = {+2, -2};
  double rho0[2] = {3.0, 5.0};

  opts.valency[0] = valency[0];
  opts.valency[1] = valency[1];

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  psi_create(pe, cs, &opts, &psi);

  /* Set the charge density, and compute an ionic strength */
  {
    int index = 2;
    int strength = 0.0;
    for (int n = 0; n < psi->nk; n++) {
      psi_rho_set(psi, index, n, rho0[n]);
      strength += 0.5*valency[n]*valency[n]*rho0[n];
    }

    {
      double value = 0.0;
      psi_ionic_strength(psi, index, &value);
      assert(fabs(value - strength) < DBL_EPSILON);
    }
  }
  
  psi_free(&psi);
  cs_free(cs);

  return 0;
}
