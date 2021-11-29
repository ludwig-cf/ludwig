/*****************************************************************************
 *
 *  test_model.c
 *
 *  Unit test for the currently compiled model (D3Q15 or D3Q19).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "lb_model_s.h"
#include "tests.h"

static void test_model_constants(void);
static void test_model_velocity_set(void);

int do_test_model_distributions(pe_t * pe, cs_t * cs);
int do_test_model_halo_swap(pe_t * pe, cs_t * cs);
int do_test_model_reduced_halo_swap(pe_t * pe, cs_t * cs);
int do_test_lb_model_io(pe_t * pe, cs_t * cs);
static  int test_model_is_domain(cs_t * cs, int ic, int jc, int kc);

/*****************************************************************************
 *
 *  test_model_suite
 *
 *****************************************************************************/

int test_model_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  /* Test model structure (coordinate-independent stuff) */

  test_model_constants();
  test_model_velocity_set();

  /* Now test actual distributions */

  do_test_model_distributions(pe, cs);
  do_test_model_halo_swap(pe, cs);
  if (DATA_MODEL == DATA_MODEL_AOS && NSIMDVL == 1) {
    do_test_model_reduced_halo_swap(pe, cs);
  }
  do_test_lb_model_io(pe, cs);

  pe_info(pe, "PASS     ./unit/test_model\n");
  cs_free(cs);
  pe_free(pe);

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
  double sum;

  LB_CS2_DOUBLE(cs2);
  LB_RCS2_DOUBLE(rcs2);
  KRONECKER_DELTA_CHAR(d_);

  test_assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  /* Speed of sound */

  test_assert(fabs(rcs2 - 3.0) < TEST_DOUBLE_TOLERANCE);

  /* Checking wv[p]*q_[p][i][j]... */

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*(cv[p][i]*cv[p][j] - cs2*d_[i][j]);
      }
      test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
    }
  }

  /* Checking wv[p]*cv[p][i]*q_[p][j][k]... */

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      for (k = 0; k < NDIM; k++) {
	sum = 0.0;
	for (p = 0; p < NVEL; p++) {
	  sum += wv[p]*cv[p][i]*(cv[p][i]*cv[p][j] - cs2*d_[i][j]);
	}
	test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
      }
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

int do_test_model_distributions(pe_t * pe, cs_t * cs) {

  int i, n, p;
  int index = 1;
  int ndist = 2;
  double fvalue, fvalue_expected;
  double u[3];

  lb_t * lb;

  assert(pe);
  assert(cs);

  /* Tests of the basic distribution functions. */

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist(lb, &n);
  assert(n == 1); /* Default */

  lb_ndist_set(lb, ndist);
  lb_init(lb);

  /* Report the number of distributions */

  lb_ndist(lb, &n);
  assert(n == ndist);

  for (n = 0; n < ndist; n++) {
    for (p = 0; p < NVEL; p++) {
      fvalue_expected = 0.01*n + wv[p];
      lb_f_set(lb, index, p, n, fvalue_expected);
      lb_f(lb, index, p, n, &fvalue);
      assert(fabs(fvalue - fvalue_expected) < DBL_EPSILON);
    }

    /* Check zeroth moment... */

    fvalue_expected = 0.01*n*NVEL + 1.0;
    lb_0th_moment(lb, index, (lb_dist_enum_t) n, &fvalue);
    assert(fabs(fvalue - fvalue_expected) <= DBL_EPSILON);

    /* Check first moment... */

    lb_1st_moment(lb, index, (n == 0) ? LB_RHO : LB_PHI, u);

    for (i = 0; i < NDIM; i++) {
      assert(fabs(u[i] - 0.0) < DBL_EPSILON);
    }
  }

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

int do_test_model_halo_swap(pe_t * pe, cs_t * cs) {

  int i, j, k, p;
  int n, ndist = 2;
  int index, nlocal[3];
  const int nextra = 1;  /* Distribution halo width always 1 */
  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);

  cs_nlocal(cs, nlocal);

  /* The test relies on a uniform decomposition in parallel:
   *
   * f[0] or f[X] is set to local x index,
   * f[1] or f[Y] is set to local y index
   * f[2] or f[Z] is set to local z index
   * remainder are set to velocity index. */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {

	index = cs_index(cs, i, j, k);

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

  lb_memcpy(lb, tdpMemcpyHostToDevice);
  lb_halo(lb);
  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  /* Test all the sites not in the interior */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = cs_index(cs, i, j, k);

	for (n = 0; n < ndist; n++) {

	  f_expect = 1.0*abs(i - nlocal[X]);
	  lb_f(lb, index, X, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = 1.0*abs(j - nlocal[Y]);
	  lb_f(lb, index, Y, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = 1.0*abs(k - nlocal[Z]);
	  lb_f(lb, index, Z, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  for (p = 3; p < NVEL; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = (double) p;
	    test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_model_reduced_halo_swap
 *
 *****************************************************************************/

int do_test_model_reduced_halo_swap(pe_t * pe, cs_t * cs) {  

  int i, j, k, p;
  int icdt, jcdt, kcdt;
  int index, nlocal[3];
  int n, ndist = 2;
  const int nextra = 1;

  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, LB_HALO_REDUCED);

  cs_nlocal(cs, nlocal);

  /* Set everything which is NOT in a halo */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    f_expect = 1.0*(n*NVEL + p);
	    lb_f_set(lb, index, p, n, f_expect);
	  }
	}
      }
    }
  }

  lb_halo_via_struct(lb);

  /* Now check that the interior sites are unchanged */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*NVEL +  p);
	    test_assert(fabs(f_expect - f_actual) < DBL_EPSILON);
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

	index = cs_index(cs, i, j, k);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {

	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*NVEL + p);

	    icdt = i + cv[p][X];
	    jcdt = j + cv[p][Y];
	    kcdt = k + cv[p][Z];

	    if (test_model_is_domain(cs, icdt, jcdt, kcdt)) {
	      test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	    }
	  }
	}

	/* Next site */
      }
    }
  }

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

static int test_model_is_domain(cs_t * cs, int ic, int jc, int kc) {

  int nlocal[3];
  int iam = 1;

  assert(cs);

  cs_nlocal(cs, nlocal);

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

int do_test_lb_model_io(pe_t * pe, cs_t * cs) {

  int ndist = 2;
  lb_t * lbrd = NULL;
  lb_t * lbwr = NULL;

  assert(pe);
  assert(cs);

  lb_create_ndist(pe, cs, ndist, &lbrd);
  lb_create_ndist(pe, cs, ndist, &lbwr);

  lb_init(lbwr);
  lb_init(lbrd);

  /* Write */

  /* Read */

  /* Compare */

  lb_free(lbwr);
  lb_free(lbrd);

  return 0;
}
