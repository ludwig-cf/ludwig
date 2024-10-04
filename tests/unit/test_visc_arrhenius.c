/*****************************************************************************
 *
 *  test_visc_arrhenius.c
 *
 *  Unit tests for Arrhenius viscosity model
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field_phi_init.h"
#include "visc_arrhenius.h"
#include "tests.h"

int test_visc_arrhenius_create(pe_t * pe, cs_t * cs, field_t * phi);
int test_visc_arrhenius_update(pe_t * pe, cs_t * cs, field_t * phi);
int test_visc_arrhenius_eta_uniform(cs_t * cs, hydro_t * hydro, double eta0);


/*****************************************************************************
 *
 *  test_visc_arrhenius_suite
 *
 *****************************************************************************/

__host__ int test_visc_arrhenius_suite(void) {

  const int nhalo = 0;

  int ndevice;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  field_t * phi = NULL;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_visc_arrhenius\n");
  }
  else {

    field_options_t opts = field_options_ndata_nhalo(1, nhalo);

    cs_create(pe, &cs);
    cs_init(cs);

    field_create(pe, cs, NULL, "ternary", &opts, &phi);

    test_visc_arrhenius_create(pe, cs, phi);
    test_visc_arrhenius_update(pe, cs, phi);

    field_free(phi);
    cs_free(cs);
  }

  pe_info(pe, "PASS     ./unit/test_visc_arrhenius\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_visc_arrhenius_create
 *
 *****************************************************************************/

int test_visc_arrhenius_create(pe_t * pe, cs_t * cs, field_t * phi) {

  visc_arrhenius_param_t param = {1.0, 2.0, 3.0};
  visc_arrhenius_t * visc = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  visc_arrhenius_create(pe, cs, phi, param, &visc);

  assert(visc);
  assert(visc->pe == pe);
  assert(visc->cs == cs);
  assert(visc->phi == phi);
  assert(visc->param);

  assert(fabs(visc->param->eta_plus - param.eta_plus)   < DBL_EPSILON);
  assert(fabs(visc->param->eta_minus - param.eta_minus) < DBL_EPSILON);
  assert(fabs(visc->param->phistar - param.phistar)     < DBL_EPSILON);

  visc_arrhenius_free(visc);

  return 0;
}

/*****************************************************************************
 *
 *  test_visc_arrhenius_update
 *
 *  Set phi = phi0 everywhere and check the resulting viscosity is
 *  consistent.
 *
 *****************************************************************************/

int test_visc_arrhenius_update(pe_t * pe, cs_t * cs, field_t * phi) {

  const double eta_plus  = 0.5;
  const double eta_minus = 0.1;
  const double phistar   = 1.0;
  
  visc_arrhenius_param_t param = {eta_minus, eta_plus, phistar};
  visc_arrhenius_t * visc = NULL;

  hydro_options_t hopts = hydro_options_nhalo(0);
  hydro_t * hydro = NULL;

  int ifail;
  double phi0;
  double eta0;

  assert(pe);
  assert(cs);
  assert(phi);

  visc_arrhenius_create(pe, cs, phi, param, &visc);

  /* Initialise the field */

  phi0 = +1.0;
  field_phi_init_uniform(phi, phi0);
  field_memcpy(phi, tdpMemcpyHostToDevice);

  /* Initialise the hydrodynamic sector and update the viscosity */

  hydro_create(pe, cs, NULL, &hopts, &hydro);
  visc_arrhenius_update(visc, hydro);
  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);

  /* Check */

  eta0 = pow(eta_minus, 0.5*(1.0-phi0))*pow(eta_plus, 0.5*(1.0+phi0));
  assert(fabs(eta0 - param.eta_plus) < DBL_EPSILON);

  ifail = test_visc_arrhenius_eta_uniform(cs, hydro, eta0);
  if (ifail) printf("test_visc_arrhenius_eta_uniform()\n");
  assert(ifail == 0);

  /* Another value phi = -1 */

  phi0 = -1.0;
  field_phi_init_uniform(phi, phi0);
  field_memcpy(phi, tdpMemcpyHostToDevice);

  visc_arrhenius_update(visc, hydro);
  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);

  /* Check eta(phi = -1) */

  eta0 = pow(eta_minus, 0.5*(1.0-phi0))*pow(eta_plus, 0.5*(1.0+phi0));
  assert(fabs(eta0 - param.eta_minus) < DBL_EPSILON);

  ifail = test_visc_arrhenius_eta_uniform(cs, hydro, eta0);
  assert(ifail == 0);

  /* Finally phi0 = 0 */

  phi0 = 0.0;
  field_phi_init_uniform(phi, phi0);
  field_memcpy(phi, tdpMemcpyHostToDevice);

  visc_arrhenius_update(visc, hydro);
  hydro_memcpy(hydro, tdpMemcpyDeviceToHost);

  eta0 = pow(eta_minus, 0.5*(1.0-phi0))*pow(eta_plus, 0.5*(1.0+phi0));
  assert(fabs(eta0 - sqrt(eta_plus)*sqrt(eta_minus)) < DBL_EPSILON);

  ifail = test_visc_arrhenius_eta_uniform(cs, hydro, eta0);
  assert(ifail == 0);

  /* Clean up */

  hydro_free(hydro);
  visc_arrhenius_free(visc);

  return 0;
}

/*****************************************************************************
 *
 *  test_visc_arrhenius_eta_uniform
 *
 *****************************************************************************/

int test_visc_arrhenius_eta_uniform(cs_t * cs, hydro_t * hydro, double eta0) {

  int ifail = 0;
  int nlocal[3];
  int ic, jc, kc, index;

  double eta;

  assert(cs);
  assert(hydro);

  cs_nlocal(cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	eta = hydro->eta->data[addr_rank0(hydro->nsite, index)];

	if (fabs(eta - eta0) > DBL_EPSILON) ifail = 1;
      }
    }
  }

  return ifail;
}
