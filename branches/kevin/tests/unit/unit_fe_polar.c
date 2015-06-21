/*****************************************************************************
 *
 *  unit_fe_polar.c
 *
 *  Unit test for polar free energy.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "coords.h"
#include "fe_polar.h"
#include "unit_control.h"

int do_test_fe_polar_param(control_t * ctrl);
int do_test_fe_polar_bulk(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_polar
 *
 *****************************************************************************/

int do_ut_fe_polar(control_t * ctrl) {

  assert(ctrl);

  do_test_fe_polar_param(ctrl);
  do_test_fe_polar_bulk(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_polar_param
 *
 *****************************************************************************/

int do_test_fe_polar_param(control_t * ctrl) {

  fe_polar_param_t param0 = {-0.1, +0.1, 0.0, 0.01, 0.00, 0.03, 0.04};
  fe_polar_param_t param1;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * p = NULL;
  field_grad_t * dp = NULL;

  fe_polar_t * fep = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Polar free energy parameter test\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 3, "p", &p);
  field_grad_create(p, 1, &dp);

  fe_polar_create(p, dp, &fep);

  try {
    fe_polar_param_set(fep, param0);
    fe_polar_param(fep, &param1);

    control_verb(ctrl, "Paramter a:         %22.15e %22.15e\n",
		 param0.a, param1.a);
    control_macro_test_dbl_eq(ctrl, param0.a, param1.a, DBL_EPSILON);
    control_verb(ctrl, "Paramter b:         %22.15e %22.15e\n",
		 param0.b, param1.b);
    control_macro_test_dbl_eq(ctrl, param0.b, param1.b, DBL_EPSILON);
    control_verb(ctrl, "Paramter delta:     %22.15e %22.15e\n",
		 param0.delta, param1.delta);
    control_macro_test_dbl_eq(ctrl, param0.delta, param1.delta, DBL_EPSILON);
    control_verb(ctrl, "Paramter kappa1:    %22.15e %22.15e\n",
		 param0.kappa1, param1.kappa1);
    control_macro_test_dbl_eq(ctrl, param0.kappa1, param1.kappa1, DBL_EPSILON);
    control_verb(ctrl, "Paramter kappa2:    %22.15e %22.15e\n",
		 param0.kappa2, param1.kappa2);
    control_macro_test_dbl_eq(ctrl, param0.kappa2, param1.kappa2, DBL_EPSILON);
    control_verb(ctrl, "Paramter lambda:    %22.15e %22.15e\n",
		 param0.lambda, param1.lambda);
    control_macro_test_dbl_eq(ctrl, param0.lambda, param1.lambda, DBL_EPSILON);
    control_verb(ctrl, "Paramter zeta:      %22.15e %22.15e\n",
		 param0.zeta, param1.zeta);
    control_macro_test_dbl_eq(ctrl, param0.zeta, param1.zeta, DBL_EPSILON);

  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  fe_polar_free(fep);

  field_grad_free(dp);
  field_free(p);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_polar_bulk
 *
 *****************************************************************************/

int do_test_fe_polar_bulk(control_t * ctrl) {

  fe_polar_param_t param0 = {-0.1, +0.1, 0.0, 0.01, 0.00, 0.03, 0.04};

  int index;
  int ia, ib;
  double p0[3] = {0.5, 0.6, 0.7};
  double fed0, fed1;
  double h0[3], h1[3];
  double s0, s1[3][3];
  double p2, ph;

  pe_t * pe = NULL;
  coords_t * cs = NULL;
  field_t * p = NULL;
  field_grad_t * dp = NULL;

  fe_polar_t * fep = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Polar bulk free energy test\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &cs);
  coords_commit(cs);

  field_create(cs, 3, "p", &p);
  field_init(p, 1, NULL);
  field_grad_create(p, 2, &dp);

  index = coords_index(cs, 1, 1, 1);
  field_vector_set(p, index, p0);

  fe_polar_create(p, dp, &fep);

  try {
    fe_polar_param_set(fep, param0);

    p2 = p0[X]*p0[X] + p0[Y]*p0[Y] + p0[Z]*p0[Z];
    fed0 = 0.5*param0.a*p2 + 0.25*param0.b*p2*p2;
    h0[X] = -param0.a*p0[X] - param0.b*p2*p0[X];
    h0[Y] = -param0.a*p0[Y] - param0.b*p2*p0[Y];
    h0[Z] = -param0.a*p0[Z] - param0.b*p2*p0[Z];

    /* Abstract interface */

    fe_fed((fe_t *) fep, index, &fed1);
    fe_hvector((fe_t *) fep, index, h1);
    fe_str((fe_t *) fep, index, s1);

    control_verb(ctrl, "fe density:         %22.15e %22.15e\n", fed0, fed1);
    control_macro_test_dbl_eq(ctrl, fed0, fed1, DBL_EPSILON);
    control_verb(ctrl, "molecular field[X]: %22.15e %22.15e\n", h0[X], h1[X]);
    control_macro_test_dbl_eq(ctrl, h0[X], h1[X], DBL_EPSILON);
    control_verb(ctrl, "molecular field[Y]: %22.15e %22.15e\n", h0[Y], h1[Y]);
    control_macro_test_dbl_eq(ctrl, h0[Y], h1[Y], DBL_EPSILON);
    control_verb(ctrl, "molecular field[Z]: %22.15e %22.15e\n", h0[Z], h1[Z]);
    control_macro_test_dbl_eq(ctrl, h0[Z], h1[Z], DBL_EPSILON);

    ph = p0[X]*h0[X] + p0[Y]*h0[Y] + p0[Z]*h0[Z];

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	double dij = (ia == ib);
	s0 = 0.5*(p0[ia]*h0[ia] - p0[ia]*h0[ia])
	  - param0.lambda*(0.5*(p0[ia]*h0[ib] + p0[ib]*h0[ia]) - dij*ph/3.0)
	  - param0.zeta*(p0[ia]*p0[ib] - dij*p2/3.0);
	control_verb(ctrl, "chem stress[%d][%d]:  %22.15e %22.15e\n", ia, ib,
		     s0, -s1[ia][ib]);
	control_macro_test_dbl_eq(ctrl, s0, -s1[ia][ib], DBL_EPSILON);
      }
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  /* Virtual destructor */
  fe_free((fe_t *) fep);

  field_grad_free(dp);
  field_free(p);
  coords_free(cs);
  pe_free(pe);

  control_report(ctrl);

  return 0;
}
