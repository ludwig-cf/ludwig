/*****************************************************************************
 *
 *  unit_fe_surf.c
 *
 *  Surfactant free energy unit test.
 *
 *****************************************************************************/

#include <assert.h>

#include "coords.h"
#include "fe_surfactant.h"
#include "unit_control.h"

int do_test_fe_surf_param(control_t * ctrl);
int do_test_fe_surf_bulk(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_surf
 *
 *****************************************************************************/

int do_ut_fe_surf(control_t * ctrl) {

  assert(ctrl);

  do_test_fe_surf_param(ctrl);
  do_test_fe_surf_bulk(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_surf_param
 *
 *****************************************************************************/

int do_test_fe_surf_param(control_t * ctrl) {

  fe_surfactant_param_t param0 = {-0.0208333, +0.0208333, +0.12,
				  0.00056587, +0.03, 0.0, 0.0};
  fe_surfactant_param_t param1;
  double sigma0, sigma1;
  double xi0, xi1;
  double psic0, psic1;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * surf = NULL;
  field_grad_t * grad = NULL;
  fe_t * fe = NULL;
  fe_surfactant_t * fs = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Surfactant free energy parameter test\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 2, "pair", &surf);
  field_grad_create(surf, 0, &grad);

  fe_create(&fe);
  fe_surfactant_create(fe, surf, grad, &fs);

  try {
    fe_surfactant_param_set(fs, param0);
    fe_surfactant_param(fs, &param1);

    control_verb(ctrl, "Parameter a:       %22.15e %22.15e\n",
		 param0.a, param1.a);
    control_macro_test_dbl_eq(ctrl, param0.a, param1.a, DBL_EPSILON);

    control_verb(ctrl, "Parameter b:       %22.15e %22.15e\n",
		 param0.b, param1.b);
    control_macro_test_dbl_eq(ctrl, param0.b, param1.b, DBL_EPSILON);

    control_verb(ctrl, "Parameter kappa:   %22.15e %22.15e\n",
		 param0.kappa, param1.kappa);
    control_macro_test_dbl_eq(ctrl, param0.kappa, param1.kappa, DBL_EPSILON);

    control_verb(ctrl, "Parameter kt:      %22.15e %22.15e\n",
		 param0.kt, param1.kt);
    control_macro_test_dbl_eq(ctrl, param0.kt, param1.kt, DBL_EPSILON);

    control_verb(ctrl, "Parameter epsilon: %22.15e %22.15e\n",
		 param0.epsilon, param1.epsilon);
    control_macro_test_dbl_eq(ctrl, param0.epsilon, param1.epsilon, DBL_EPSILON);

    control_verb(ctrl, "Parameter beta:    %22.15e %22.15e\n",
		 param0.beta, param1.beta);
    control_macro_test_dbl_eq(ctrl, param0.beta, param1.beta, DBL_EPSILON);

    control_verb(ctrl, "Parameter w:       %22.15e %22.15e\n",
		 param0.w, param1.w);
    control_macro_test_dbl_eq(ctrl, param0.w, param1.w, DBL_EPSILON);

    /* Derived quantities */

    fe_surfactant_sigma(fs, &sigma0);
    fe_surfactant_xi0(fs, &xi0);
    fe_surfactant_langmuir_isotherm(fs, &psic0);

    sigma1 = sqrt(-8.0*param0.kappa*param0.a/9.0);
    xi1 = sqrt(-2.0*param0.kappa/param0.a);
    psic1 = exp(param0.epsilon/(2.0*param0.kt*xi1*xi1));

    control_verb(ctrl, "Interface sigma:   %22.15e %22.15e\n", sigma0, sigma1);
    control_macro_test_dbl_eq(ctrl, sigma0, sigma1, DBL_EPSILON);

    control_verb(ctrl, "Interface width:   %22.15e %22.15e\n", xi0, xi1);
    control_macro_test_dbl_eq(ctrl, xi0, xi1, DBL_EPSILON);

    control_verb(ctrl, "Langmuir isotherm: %22.15e %22.15e\n", psic0, psic1);
    control_macro_test_dbl_eq(ctrl, psic0, psic1, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_surfactant_free(fs);
  fe_free(fe);

  field_grad_free(grad);
  field_free(surf);

  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_surf_bulk
 *
 *****************************************************************************/

int do_test_fe_surf_bulk(control_t * ctrl) {

  fe_surfactant_param_t param0 = {-0.0208333, +0.0208333, +0.12,
    0.00056587, +0.03, 0.0, 0.0};
  /*fe_surfactant_param_t param0 = {-0.0208333, +0.0208333, 0.0,
    0.00056587, 0.0, 0.0, 0.0};*/
  int index;
  double f[2] = {0.5, 0.01};  /* phi, psi */
  double fed0, fed1;
  double mu0[2], mu1[2];

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * surf = NULL;
  field_grad_t * grad = NULL;
  fe_t * fe = NULL;
  fe_surfactant_t * fs = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Surfactant bulk free energy\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 2, "pair", &surf);
  field_init(surf, 1, NULL);
  field_grad_create(surf, 2, &grad);

  index = coords_index(cs, 1, 1, 1);
  field_scalar_array_set(surf, index, f);

  fe_create(&fe);
  fe_surfactant_create(fe, surf, grad, &fs);

  try {
    fe_surfactant_param_set(fs, param0);

    fe_fed(fe, index, &fed0);
    fe_mu(fe, index, mu0);

    fed1 = 0.5*(param0.a + 0.5*param0.b*f[0]*f[0])*f[0]*f[0]
      + param0.kt*(f[1]*log(f[1]) + (1.0 - f[1])*log(1.0 - f[1]));
    mu1[0] = param0.a*f[0] + param0.b*f[0]*f[0]*f[0];
    mu1[1] = param0.kt*(log(f[1]) - log(1.0 - f[1]));

    control_verb(ctrl, "fe density:         %22.15e %22.15e\n", fed0, fed1);
    control_macro_test_dbl_eq(ctrl, fed0, fed1, DBL_EPSILON);

    control_verb(ctrl, "mu phi:             %22.15e %22.15e\n", mu0[0], mu1[0]);
    control_macro_test_dbl_eq(ctrl, mu0[0], mu1[0], DBL_EPSILON);

    control_verb(ctrl, "mu psi:             %22.15e %22.15e\n", mu0[1], mu1[1]);
    control_macro_test_dbl_eq(ctrl, mu0[1], mu1[1], DBL_EPSILON);


  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_surfactant_free(fs);
  fe_free(fe);

  field_grad_free(grad);
  field_free(surf);

  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}
