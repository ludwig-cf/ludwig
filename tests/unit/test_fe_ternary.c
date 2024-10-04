/*****************************************************************************
 *
 *  test_fe_tenery.c
 *
 *  Unit tests for ternary free energy.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Shan Chen (shan.chen@epfl.ch)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "fe_ternary.h"
#include "tests.h"

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_fed(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_mu(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_ternary_str(pe_t * pe, cs_t * cs, field_t * phi);

/*****************************************************************************
 *
 *  test_fe_tenrary_suite
 *
 *****************************************************************************/

__host__ int test_fe_ternary_suite(void) {

  const int nf2 = 2;
  const int nhalo = 2;

  int ndevice;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  field_t * phi = NULL;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_fe_ternary\n");
  }
  else {

    field_options_t opts = field_options_ndata_nhalo(nf2, nhalo);

    cs_create(pe, &cs);
    cs_init(cs);

    field_create(pe, cs, NULL, "ternary", &opts, &phi);

    test_fe_ternary_create(pe, cs, phi);
    test_fe_ternary_fed(pe, cs, phi);
    test_fe_ternary_mu(pe, cs, phi);
    test_fe_ternary_str(pe, cs, phi);

    field_free(phi);
    cs_free(cs);
  }

  pe_info(pe, "PASS     ./unit/test_fe_ternary\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_create
 *
 *****************************************************************************/

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  fe_ternary_param_t pref = {0.1, 0.2, 0.3, 0.4};
  fe_ternary_param_t p    = {0};
  field_grad_t * dphi = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  assert(fe);

  fe_ternary_param(fe, &p);
  test_assert((pref.alpha  - p.alpha)  < DBL_EPSILON);
  test_assert((pref.kappa1 - p.kappa1) < DBL_EPSILON);
  test_assert((pref.kappa2 - p.kappa2) < DBL_EPSILON);
  test_assert((pref.kappa3 - p.kappa3) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_fed
 *
 *****************************************************************************/

__host__ int test_fe_ternary_fed(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  const double grad[2][3] = {{0.1, -0.2, 0.3}, {-0.4, 0.5, -0.7}};
  double fed;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_fed(fe, index, &fed);
  test_assert(fabs(fed - 3.3075000e-02) < DBL_EPSILON);

  /* With gradients */

  field_grad_pair_grad_set(dphi, index, grad);
  fe_ternary_fed(fe, index, &fed);
  test_assert(fabs(fed - 1.6313750e-01) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_mu
 *
 *****************************************************************************/

__host__ int test_fe_ternary_mu(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  const double d2phi[2] = {0.1, 0.4};
  double mu[3];

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_mu(fe, index, mu);
  test_assert(fabs(mu[0] - -2.9400000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[1] - -9.6600000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[2] - -2.9400000e-02) < DBL_EPSILON);

  /* With delsq */

  field_grad_pair_delsq_set(dphi, index, d2phi);
  fe_ternary_mu(fe, index, mu);
  test_assert(fabs(mu[0] - -4.0025000e-02) < DBL_EPSILON);
  test_assert(fabs(mu[1] - -2.0972500e-01) < DBL_EPSILON);
  test_assert(fabs(mu[2] - -5.0250000e-03) < DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_str
 *
 *****************************************************************************/

__host__ int test_fe_ternary_str(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  field_grad_t * dphi = NULL;
  fe_ternary_param_t pref = {0.5, 0.6, 0.7, 0.8};

  int index = 1;
  double phi0[2] = {-0.3, 0.7};
  double d2phi[2] = {0.1, 0.4};
  const double grad[2][3] = {{0.1, -0.2, 0.3}, {-0.4, 0.5, -0.7}};
  double s[3][3];

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients */

  field_scalar_array_set(phi, index, phi0);
  fe_ternary_str(fe, index, s);

  /* DBL_EPSILON is just too tight for some platform/compiler combinations */

  test_assert(fabs(s[0][0] - 5.2552500e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][1] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][2] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][0] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][1] - 5.2552500e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][2] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][0] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][1] - 0.0000000e+00) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][2] - 5.2552500e-01) < 2.0*DBL_EPSILON);

  /* With grad */

  field_grad_pair_grad_set(dphi, index, grad);
  fe_ternary_str(fe, index, s);

  test_assert(fabs(s[0][0] -  4.4077500e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][1] - -5.7062500e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][2] -  8.0000000e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][0] - -5.7062500e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][1] -  4.6777500e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][2] - -1.0150000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][0] -  8.0000000e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][1] - -1.0150000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][2] -  5.3796250e-01) < 2.0*DBL_EPSILON);

  /* With delsq */

  field_grad_pair_delsq_set(dphi, index, d2phi);
  fe_ternary_str(fe, index, s);

  test_assert(fabs(s[0][0] -  3.9790000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][1] - -5.7062500e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[0][2] -  8.0000000e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][0] - -5.7062500e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][1] -  4.2490000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[1][2] - -1.0150000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][0] -  8.0000000e-02) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][1] - -1.0150000e-01) < 2.0*DBL_EPSILON);
  test_assert(fabs(s[2][2] -  4.9508750e-01) < 2.0*DBL_EPSILON);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

