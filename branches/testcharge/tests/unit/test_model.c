/*****************************************************************************
 *
 *  test_model.c
 *
 *  Unit test for the currently compiled model (D3Q15 or D3Q19).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "model.h"
#include "tests.h"

static void test_model_constants(void);
static void test_model_velocity_set(void);

int do_test_model_distributions(void);
int do_test_model_halo_swap(void);
int do_test_model_reduced_halo_swap(void);
int do_test_lb_model_io(void);
static  int test_model_is_domain(const int ic, const int jc, const int kc);

/*****************************************************************************
 *
 *  test_model_suite
 *
 *****************************************************************************/

int test_model_suite(void) {

  pe_init_quiet();
  coords_init();

  /* Test model structure (coordinate-independent stuff) */

  test_model_constants();
  test_model_velocity_set();

  /* Now test actual distributions */

  do_test_model_distributions();
  do_test_model_halo_swap();
  do_test_model_reduced_halo_swap();
  do_test_lb_model_io();

  info("PASS     ./unit/test_model\n");
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_model_constants
 *
 *  Check the various constants associated with the reduced halo swap.
 *
 *****************************************************************************/

static void test_model_constants(void) {

  int i, k, p;

  for (i = 0; i < CVXBLOCK; i++) {
    for (k = 0; k < xblocklen_cv[i]; k++) {
      p = xdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == +1);
      p = xdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == -1);
    }
  }

  for (i = 0; i < CVYBLOCK; i++) {
    for (k = 0; k < yblocklen_cv[i]; k++) {
      p = ydisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == +1);
      p = ydisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == -1);
    }
  }

  for (i = 0; i < CVZBLOCK; i++) {
    for (k = 0; k < zblocklen_cv[i]; k++) {
      p = zdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == +1);
      p = zdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == -1);
    }
  }

  /* info("Model constants ok.\n"); */

  return;
}

/*****************************************************************************
 *
 *  test_model_velocity_set
 *
 *  Check the velocities, kinetic projector, tables of eigenvectors
 *  etc etc are all consistent for the current model.
 *
 *****************************************************************************/

static void test_model_velocity_set(void) {

  int i, j, k, p;
  double dij;
  double sum, sumx, sumy, sumz;

  /*
  info("Checking velocities cv etc...\n\n");

  info("The number of dimensions appears to be NDIM = %d\n", NDIM);
  info("The model appears to have NVEL = %d\n", NVEL);
  info("Number of hydrodynamic modes: %d\n", 1 + NDIM + NDIM*(NDIM+1)/2);
  */

  test_assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  /* Speed of sound */

  /* info("The speed of sound is 1/3... ");*/
  test_assert(fabs(rcs2 - 3.0) < TEST_DOUBLE_TOLERANCE);

  /* Kronecker delta */

  /* info("Checking Kronecker delta d_ij...");*/

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      if (i == j) {
	test_assert(fabs(d_[i][j] - 1.0) < TEST_DOUBLE_TOLERANCE);
      }
      else {
	test_assert(fabs(d_[i][j] - 0.0) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* info("Checking cv[0][X] = 0 etc...");*/

  test_assert(cv[0][X] == 0);
  test_assert(cv[0][Y] == 0);
  test_assert(cv[0][Z] == 0);

  /* info("Checking cv[p][X] = -cv[NVEL-p][X] (p != 0) etc..."); */

  for (p = 1; p < NVEL; p++) {
    test_assert(cv[p][X] == -cv[NVEL-p][X]);
    test_assert(cv[p][Y] == -cv[NVEL-p][Y]);
    test_assert(cv[p][Z] == -cv[NVEL-p][Z]);
  }

  /* Sum of quadrature weights, velcoities */

  /* info("Checking sum of wv[p]... ");*/

  sum = 0.0; sumx = 0.0; sumy = 0.0; sumz = 0.0;

  for (p = 0; p < NVEL; p++) {
    sum += wv[p];
    sumx += wv[p]*cv[p][X];
    sumy += wv[p]*cv[p][Y];
    sumz += wv[p]*cv[p][Z];
  }

  test_assert(fabs(sum - 1.0) < TEST_DOUBLE_TOLERANCE);
  /* info("Checking sum of wv[p]*cv[p][X]... "); */
  test_assert(fabs(sumx) < TEST_DOUBLE_TOLERANCE);
  /* info("Checking sum of wv[p]*cv[p][Y]... "); */
  test_assert(fabs(sumy) < TEST_DOUBLE_TOLERANCE);
  /* info("Checking sum of wv[p]*cv[p][Z]... "); */
  test_assert(fabs(sumz) < TEST_DOUBLE_TOLERANCE);

  /* Quadratic terms = cs^2 d_ij */

  /* info("Checking wv[p]*cv[p][i]*cv[p][j]...");*/

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*cv[p][i]*cv[p][j];
      }
      test_assert(fabs(sum - d_[i][j]/rcs2) < TEST_DOUBLE_TOLERANCE);
    }
  }

  /* info("Checking q_[p][i][j] = cv[p][i]*cv[p][j] - c_s^2*d_[i][j]...");*/

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < NDIM; i++) {
      for (j = 0; j < NDIM; j++) {
	sum = cv[p][i]*cv[p][j] - d_[i][j]/rcs2;
	test_assert(fabs(sum - q_[p][i][j]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* info("Checking wv[p]*q_[p][i][j]...");*/

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*q_[p][i][j];
      }
      test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
    }
  }

  /* info("Checking wv[p]*cv[p][i]*q_[p][j][k]...");*/

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      for (k = 0; k < NDIM; k++) {
	sum = 0.0;
	for (p = 0; p < NVEL; p++) {
	  sum += wv[p]*cv[p][i]*q_[p][j][k];
	}
	test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* No actual test here yet. Requires a theoretical answer. */
  /* info("Checking d_[i][j]*q_[p][i][j]...");*/

  for (p = 0; p < NVEL; p++) {
    sum = 0.0;
    for (i = 0; i < NDIM; i++) {
      for (j = 0; j < NDIM; j++) {
	sum += d_[i][j]*q_[p][i][j];
      }
    }
    /* test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);*/
    /* info("p = %d sum = %f\n", p, sum);*/
  }

  /* info("Check ma_ against rho, cv ... ");*/

  for (p = 0; p < NVEL; p++) {
    test_assert(fabs(ma_[0][p] - 1.0) < TEST_DOUBLE_TOLERANCE);
    for (i = 0; i < NDIM; i++) {
      test_assert(fabs(ma_[1+i][p] - cv[p][i]) < TEST_DOUBLE_TOLERANCE);
    }
  }

  /* info("Check ma_ against q_ ...");*/

  for (p = 0; p < NVEL; p++) {
    k = 0;
    for (i = 0; i < NDIM; i++) {
      for (j = i; j < NDIM; j++) {
	test_assert(fabs(ma_[1 + NDIM + k++][p] - q_[p][i][j])
		    < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* info("Checking normalisers norm_[i]*wv[p]*ma_[i][p]*ma_[j][p] = dij.");*/

  for (i = 0; i < NVEL; i++) {
    for (j = 0; j < NVEL; j++) {
      dij = (double) (i == j);
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += norm_[i]*wv[p]*ma_[i][p]*ma_[j][p];
      }
      test_assert(fabs(sum - dij) < TEST_DOUBLE_TOLERANCE);
    }
  }

  /* info("Checking ma_[i][p]*mi_[p][j] = dij ... ");*/

  for (i = 0; i < NVEL; i++) {
    for (j = 0; j < NVEL; j++) {
      dij = (double) (i == j);
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += ma_[i][p]*mi_[p][j];
      }
      test_assert(fabs(sum - dij) < TEST_DOUBLE_TOLERANCE);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  do_test_model_distributions
 *
 *  Test the distribution interface.
 *
 *****************************************************************************/

int do_test_model_distributions(void) {

  int i, n, p;
  int index = 1;
  int ndist = 2;
  double fvalue, fvalue_expected;
  double u[3];

  lb_t * lb;

  /* Tests of the basic distribution functions. */

  lb_create(&lb);
  assert(lb);
  lb_ndist(lb, &n);
  assert(n == 1); /* Default */

  lb_ndist_set(lb, ndist);
  lb_init(lb);

  /* Report the number of distributions */

  lb_ndist(lb, &n);
  test_assert(n == ndist);

  for (n = 0; n < ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      fvalue_expected = 0.01*n + wv[p];
      lb_f_set(lb, index, p, n, fvalue_expected);
      lb_f(lb, index, p, n, &fvalue);
      assert(fabs(fvalue - fvalue_expected) < DBL_EPSILON);
    }

    /* info("Check zeroth moment... ");*/

    fvalue_expected = 0.01*n*NVEL + 1.0;
    lb_0th_moment(lb, index, (lb_dist_enum_t) n, &fvalue);
    assert(fabs(fvalue - fvalue_expected) <= DBL_EPSILON);

    /* info("Check first moment... ");*/

    lb_1st_moment(lb, index, n, u);

    for (i = 0; i < NDIM; i++) {
      assert(fabs(u[i] - 0.0) < DBL_EPSILON);
    }
  }

  /* info("passed 1st lb check\n");*/
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_model_halo_swap
 *
 *  Test full halo swap.
 *
 *****************************************************************************/

int do_test_model_halo_swap() {

  int i, j, k, p;
  int n, ndist = 2;
  int index, nlocal[3];
  const int nextra = 1;  /* Distribution halo width always 1 */
  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);

  coords_nlocal(nlocal);

  /* The test relies on a uniform decomposition in parallel:
   *
   * f[0] or f[X] is set to local x index,
   * f[1] or f[Y] is set to local y index
   * f[2] or f[Z] is set to local z index
   * remainder are set to velocity index. */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {

	index = coords_index(i, j, k);

	for (n = 0; n < ndist; n++) {
	  lb_f_set(lb, index, X, n, (double) (i));
	  lb_f_set(lb, index, Y, n, (double) (j));
	  lb_f_set(lb, index, Z, n, (double) (k));

	  for (p = 3; p < NVEL; p++) {
	    lb_f_set(lb, index, p, n, (double) p);
	  }
	}
      }
    }
  }

  lb_halo(lb);

  /* Test all the sites not in the interior */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = coords_index(i, j, k);

	for (n = 0; n < ndist; n++) {

	  f_expect = fabs(i - nlocal[X]);
	  lb_f(lb, index, X, n, &f_actual);
	  assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = fabs(j - nlocal[Y]);
	  lb_f(lb, index, Y, n, &f_actual);
	  assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = fabs(k - nlocal[Z]);
	  lb_f(lb, index, Z, n, &f_actual);
	  assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  for (p = 3; p < NVEL; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = (double) p;
	    assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  /* info("lb check halo swap\n");*/
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_model_reduced_halo_swap
 *
 *****************************************************************************/

int do_test_model_reduced_halo_swap(void) {  

  int i, j, k, p;
  int icdt, jcdt, kcdt;
  int index, nlocal[3];
  int n, ndist = 2;
  const int nextra = 1;

  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  lb_create(&lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, LB_HALO_REDUCED);

  coords_nlocal(nlocal);

  /* Set everything which is NOT in a halo */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = coords_index(i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    f_expect = 1.0*(n*NVEL + p);
	    lb_f_set(lb, index, p, n, f_expect);
	  }
	}
      }
    }
  }

  lb_halo(lb);

  /* Now check that the interior sites are unchanged */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = coords_index(i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*NVEL +  p);
	    assert(fabs(f_expect - f_actual) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  /* Also check the halos sites. The key test of the reduced halo
   * swap is that distributions for which r + c_i dt takes us into
   * the domain proper must be correct. */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = coords_index(i, j, k);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {

	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*NVEL + p);

	    icdt = i + cv[p][X];
	    jcdt = j + cv[p][Y];
	    kcdt = k + cv[p][Z];

	    if (test_model_is_domain(icdt, jcdt, kcdt)) {
	      assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	    }
	  }
	}

	/* Next site */
      }
    }
  }

  /* info("lb check reduced\n");*/
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_model_is_domain
 *
 *  Is (ic, jc, kc) in the domain proper?
 *
 *****************************************************************************/

static int test_model_is_domain(const int ic, const int jc, const int kc) {

  int nlocal[3];
  int iam = 1;

  coords_nlocal(nlocal);

  if (ic < 1) iam = 0;
  if (jc < 1) iam = 0;
  if (kc < 1) iam = 0;
  if (ic > nlocal[X]) iam = 0;
  if (jc > nlocal[Y]) iam = 0;
  if (kc > nlocal[Z]) iam = 0;

  return iam;
}

/*****************************************************************************
 *
 *  do_test_lb_model_io
 *
 *****************************************************************************/

int do_test_lb_model_io(void) {

  int ndist = 2;
  lb_t * lbrd = NULL;
  lb_t * lbwr = NULL;

  lb_create_ndist(ndist, &lbrd);
  lb_create_ndist(ndist, &lbwr);

  lb_init(lbwr);
  lb_init(lbrd);
  /* lb_io_info_set(lbwr, info);*/

  /* Write */

  /* test_lb_model_write1(lbwr);*/

  /* Read */

  /* Compare */

  lb_free(lbwr);
  lb_free(lbrd);

  return 0;
}
