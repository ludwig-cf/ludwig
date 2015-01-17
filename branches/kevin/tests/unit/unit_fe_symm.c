/*****************************************************************************
 *
 *  unit_fe_symm.c
 *
 *  Unit test for symmetric free energy.
 *
 *  TODO: gradient free enetgy; remaining assertions for stress terms
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "coords.h"
#include "fe_symmetric.h"
#include "unit_control.h"

int do_test_fe_symm_param(control_t * ctrl);
int do_test_fe_symm_bulk(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_symm
 *
 *****************************************************************************/

int do_ut_fe_symm(control_t * ctrl) {

  assert(ctrl);
  do_test_fe_symm_param(ctrl);
  do_test_fe_symm_bulk(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_symm_param
 *
 *****************************************************************************/

int do_test_fe_symm_param(control_t * ctrl) {

  fe_symmetric_param_t param0 = {-0.0625, +0.0625, +0.04};
  fe_symmetric_param_t param1;
  double sigma0, sigma1;
  double xi0, xi1;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  fe_t * fe = NULL;
  fe_symmetric_t * fs = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Symmetric FE parameter test\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 1, "phi", &phi);
  field_grad_create(phi, 2, &dphi);

  fe_create(&fe);
  fe_symmetric_create(fe, phi, dphi, &fs);

  try {

    fe_symmetric_param_set(fs, param0);
    fe_symmetric_param(fs, &param1);

    control_verb(ctrl, "Parameter A:     %22.15e %22.15e\n",
		 param0.a, param1.a);
    control_macro_test_dbl_eq(ctrl, param0.a, param1.a, DBL_EPSILON);
    control_verb(ctrl, "Parameter B:     %22.15e %22.15e\n",
		 param0.b, param1.b);
    control_macro_test_dbl_eq(ctrl, param0.b, param1.b, DBL_EPSILON);
    control_verb(ctrl, "Parameter kappa: %22.15e %22.15e\n",
		 param0.kappa, param1.kappa);
    control_macro_test_dbl_eq(ctrl, param0.kappa, param1.kappa, DBL_EPSILON);

    sigma0 = sqrt(-8.0*param0.kappa*pow(param0.a, 3)/(9.0*pow(param0.b, 2)));
    fe_symmetric_interfacial_tension(fs, &sigma1);
    control_verb(ctrl, "Tension:         %22.15e %22.15e\n", sigma0, sigma1);
    control_macro_test_dbl_eq(ctrl, sigma0, sigma1, DBL_EPSILON);

    xi0 = sqrt(-2.0*param0.kappa/param0.a);
    fe_symmetric_interfacial_width(fs, &xi1);
    control_verb(ctrl, "Width:           %22.15e %22.15e\n", xi0, xi1);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_symmetric_free(fs);
  fe_free(fe);

  field_grad_free(dphi);
  field_free(phi);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_symm_bulk
 *
 *****************************************************************************/

int do_test_fe_symm_bulk(control_t * ctrl) {

  fe_symmetric_param_t param0 = {-0.003125, +0.003125, +0.002};

  int index = 1;
  int ia, ib;
  double phi0 = 0.5;
  double fed0, fed1;
  double mu0, mu1;
  double s0, s1[3][3];

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  fe_t * fe = NULL;
  fe_symmetric_t * fs = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Symmetric bulk free energy test\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 1, "phi", &phi);
  field_grad_create(phi, 2, &dphi);
  field_init(phi, 1, NULL);
  field_scalar_set(phi, index, phi0);

  fe_create(&fe);
  fe_symmetric_create(fe, phi, dphi, &fs);

  try {

    fe_symmetric_param_set(fs, param0);

    /* Free energy density, chemical potential and diagonal term in stress */

    fed0 = 0.5*(param0.a + 0.5*param0.b*phi0*phi0)*phi0*phi0;
    mu0  = (param0.a + param0.b*phi0*phi0)*phi0;
    s0   = 0.5*(param0.a + 1.5*param0.b*phi0*phi0)*phi0*phi0;

    fe_fed(fe, index, &fed1);
    fe_mu(fe, index, &mu1);
    fe_str(fe, index, s1);

    control_verb(ctrl, "Bulk fe density: %22.15e %22.15e\n", fed0, fed1);
    control_macro_test_dbl_eq(ctrl, fed0, fed1, DBL_EPSILON);

    control_verb(ctrl, "Bulk chem. pot.: %22.15e %22.15e\n", mu0, mu1);
    control_macro_test_dbl_eq(ctrl, mu0, mu1, DBL_EPSILON);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	double dab = (ia == ib);
	control_verb(ctrl, "str[%d][%d]:       %22.15e %22.15e\n",
		     ia, ib, s0*dab, s1[ia][ib]);
	control_macro_test_dbl_eq(ctrl, s0*dab, s1[ia][ib], DBL_EPSILON);
      }
    }

    /* Explicit interface */

    fe_symmetric_fed(fs, index, &fed1);
    fe_symmetric_mu(fs, index, &mu1);

    control_macro_test_dbl_eq(ctrl, fed0, fed1, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, mu0, mu1, DBL_EPSILON);

  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_symmetric_free(fs);
  fe_free(fe);

  field_grad_free(dphi);
  field_free(phi);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}
