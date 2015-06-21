/*****************************************************************************
 *
 *  unit_fe_braz.c
 *
 *  Unit test Brazovskii free energy.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "coords.h"
#include "fe_brazovskii.h"
#include "unit_control.h"

int do_test_fe_braz_param(control_t * ctrl);
int do_test_fe_braz_bulk(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_braz
 *
 *****************************************************************************/

int do_ut_fe_braz(control_t * ctrl) {

  assert(ctrl);

  do_test_fe_braz_param(ctrl);
  do_test_fe_braz_bulk(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_braz_param
 *
 *****************************************************************************/

int do_test_fe_braz_param(control_t * ctrl) {

  fe_brazovskii_param_t param0 = {-0.0005, +0.0005, +0.00076, -0.0006};
  fe_brazovskii_param_t param1;
  double a0, a1;
  double lambda0, lambda1;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  fe_brazovskii_t * fb = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Brazovskii free energy parameters\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 1, "phi", &phi);
  field_grad_create(phi, 0, &dphi);

  fe_brazovskii_create(phi, dphi, &fb);

  try {
    fe_brazovskii_param_set(fb, param0);
    fe_brazovskii_param(fb, &param1);

    control_verb(ctrl, "Paramater a:     %22.15e %22.15e\n",
		 param0.a, param1.a);
    control_macro_test_dbl_eq(ctrl, param0.a, param1.a, DBL_EPSILON);

    control_verb(ctrl, "Parameter b:     %22.15e %22.15e\n",
		 param0.b, param1.b);
    control_macro_test_dbl_eq(ctrl, param0.b, param1.b, DBL_EPSILON);

    control_verb(ctrl, "Parameter c:     %22.15e %22.15e\n",
		 param0.c, param1.c);
    control_macro_test_dbl_eq(ctrl, param0.c, param1.c, DBL_EPSILON);

    control_verb(ctrl, "Parameter kappa: %22.15e %22.15e\n",
		 param0.kappa, param1.kappa);
    control_macro_test_dbl_eq(ctrl, param0.kappa, param1.kappa, DBL_EPSILON);

    /* Derived quantities */

    a0 =
      sqrt(4.0*(1.0 + param0.kappa*param0.kappa/(4.0*param0.b*param0.c))/3.0);
    fe_brazovskii_amplitude(fb, &a1);

    control_verb(ctrl, "Amplitude:       %22.15e %22.15e\n", a0, a1);
    control_macro_test_dbl_eq(ctrl, a0, a1, DBL_EPSILON);

    lambda0 = 8.0*atan(1.0)/sqrt(-param0.kappa/(2.0*param0.c));
    fe_brazovskii_wavelength(fb, &lambda1);

    control_verb(ctrl, "Wavelength:      %22.15e %22.15e\n", lambda0, lambda1);
    control_macro_test_dbl_eq(ctrl, lambda0, lambda1, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_brazovskii_free(fb);

  field_grad_free(dphi);
  field_free(phi);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_braz_bulk
 *
 *****************************************************************************/

int do_test_fe_braz_bulk(control_t * ctrl) {

  fe_brazovskii_param_t param0 = {-0.00001, +0.00001, +0.00076, -0.0006};
  int index;
  int ia, ib;
  double phi0 = 0.5;
  double fed0, fed1;
  double mu0, mu1;
  double s0, s1[3][3];

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  fe_brazovskii_t * fb = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Brazovskii bulk free energy\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);
  index = coords_index(cs, 1, 1, 1);

  field_create(cs, 1, "phi", &phi);
  field_init(phi, 1, NULL);

  field_grad_create(phi, 4, &dphi);
  field_scalar_set(phi, index, phi0);

  fe_brazovskii_create(phi, dphi, &fb);

  try {
    fe_brazovskii_param_set(fb, param0);

    fed0 = 0.5*(param0.a + 0.5*param0.b*phi0*phi0)*phi0*phi0;
    mu0  = (param0.a + param0.b*phi0*phi0)*phi0;
    s0   = 0.5*(param0.a + 1.5*param0.b*phi0*phi0)*phi0*phi0;

    /* Via abstract free energy */

    fe_fed((fe_t *) fb, index, &fed1);
    fe_mu((fe_t *) fb, index, &mu1);
    fe_str((fe_t *) fb, index, s1);

    control_verb(ctrl, "fe density:      %22.15e %22.15e\n", fed0, fed1);
    control_macro_test_dbl_eq(ctrl, fed0, fed1, DBL_EPSILON);

    control_verb(ctrl, "chem. pot.:      %22.15e %22.15e\n", mu0, mu1);
    control_macro_test_dbl_eq(ctrl, mu0, mu1, DBL_EPSILON);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	double dab = (ia == ib);
	control_verb(ctrl, "str[%d][%d]:       %22.15e %22.15e\n",
		     ia, ib, s0*dab, s1[ia][ib]);
	control_macro_test_dbl_eq(ctrl, s0*dab, s1[ia][ib], DBL_EPSILON);
      }
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  /* Virtual destructor */
  fe_free((fe_t *) fb);

  field_grad_free(dphi);
  field_free(phi);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}
