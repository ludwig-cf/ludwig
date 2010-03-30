/*****************************************************************************
 *
 *  test_model.c
 *
 *  Unit test for the currently compiled model (D3Q15 or D3Q19).
 *
 *  $Id: test_model.c,v 1.9.2.4 2010-03-30 03:55:40 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2010)
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "model.h"
#include "tests.h"

static void test_model_constants(void);
static void test_model_velocity_set(void);
static void test_model_distributions(void);
static void test_model_halo_swap(void);
static void test_model_reduced_halo_swap(void);
static  int test_model_is_domain(const int ic, const int jc, const int kc);

int main(int argc, char ** argv) {

  pe_init(argc, argv);

  info("Testing D%1dQ%d\n", NDIM, NVEL);

  /* Test model structure (coordinate-independent stuff) */

  test_model_constants();
  test_model_velocity_set();

  /* Now test actual distributions */

  coords_init();

  test_model_distributions();
  test_model_halo_swap();
  test_model_reduced_halo_swap();

  info("\nModel tests passed ok.\n\n");

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

  info("Model constants ok.\n");

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

  info("Checking velocities cv etc...\n\n");

  info("The number of dimensions appears to be NDIM = %d\n", NDIM);
  info("The model appears to have NVEL = %d\n", NVEL);
  info("Number of hydrodynamic modes: %d\n", 1 + NDIM + NDIM*(NDIM+1)/2);
  test_assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  /* Speed of sound */

  info("The speed of sound is 1/3... ");
  test_assert(fabs(rcs2 - 3.0) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Kronecker delta */

  info("Checking Kronecker delta d_ij...");

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
  info("ok\n");

  info("Checking cv[0][X] = 0 etc...");

  test_assert(cv[0][X] == 0);
  test_assert(cv[0][Y] == 0);
  test_assert(cv[0][Z] == 0);

  info("ok\n");


  info("Checking cv[p][X] = -cv[NVEL-p][X] (p != 0) etc...");

  for (p = 1; p < NVEL; p++) {
    test_assert(cv[p][X] == -cv[NVEL-p][X]);
    test_assert(cv[p][Y] == -cv[NVEL-p][Y]);
    test_assert(cv[p][Z] == -cv[NVEL-p][Z]);
  }

  info("ok\n");

  /* Sum of quadrature weights, velcoities */

  info("Checking sum of wv[p]... ");

  sum = 0.0; sumx = 0.0; sumy = 0.0; sumz = 0.0;

  for (p = 0; p < NVEL; p++) {
    sum += wv[p];
    sumx += wv[p]*cv[p][X];
    sumy += wv[p]*cv[p][Y];
    sumz += wv[p]*cv[p][Z];
  }

  test_assert(fabs(sum - 1.0) < TEST_DOUBLE_TOLERANCE);
  info("ok\n"); 
  info("Checking sum of wv[p]*cv[p][X]... ");
  test_assert(fabs(sumx) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");
  info("Checking sum of wv[p]*cv[p][Y]... ");
  test_assert(fabs(sumy) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");
  info("Checking sum of wv[p]*cv[p][Z]... ");
  test_assert(fabs(sumz) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Quadratic terms = cs^2 d_ij */

  info("Checking wv[p]*cv[p][i]*cv[p][j]...");

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*cv[p][i]*cv[p][j];
      }
      test_assert(fabs(sum - d_[i][j]/rcs2) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");


  /* Check q_ */

  info("Checking q_[p][i][j] = cv[p][i]*cv[p][j] - c_s^2*d_[i][j]...");

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < NDIM; i++) {
      for (j = 0; j < NDIM; j++) {
	sum = cv[p][i]*cv[p][j] - d_[i][j]/rcs2;
	test_assert(fabs(sum - q_[p][i][j]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  info("ok\n");


  info("Checking wv[p]*q_[p][i][j]...");

  for (i = 0; i < NDIM; i++) {
    for (j = 0; j < NDIM; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*q_[p][i][j];
      }
      test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");

  info("Checking wv[p]*cv[p][i]*q_[p][j][k]...");

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
  info("ok\n");

  /* No actual test here yet. Requires a theoretical answer. */
  info("Checking d_[i][j]*q_[p][i][j]...");

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
  info("ok\n");


  info("Check ma_ against rho, cv ... ");

  for (p = 0; p < NVEL; p++) {
    test_assert(fabs(ma_[0][p] - 1.0) < TEST_DOUBLE_TOLERANCE);
    for (i = 0; i < NDIM; i++) {
      test_assert(fabs(ma_[1+i][p] - cv[p][i]) < TEST_DOUBLE_TOLERANCE);
    }
  }

  info("ok\n");

  info("Check ma_ against q_ ...");

  for (p = 0; p < NVEL; p++) {
    k = 0;
    for (i = 0; i < NDIM; i++) {
      for (j = i; j < NDIM; j++) {
	test_assert(fabs(ma_[1 + NDIM + k++][p] - q_[p][i][j])
		    < TEST_DOUBLE_TOLERANCE);
      }
    }
    /*
    test_assert(fabs(ma_[4][p] - q_[p][X][X]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[5][p] - q_[p][X][Y]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[6][p] - q_[p][X][Z]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[7][p] - q_[p][Y][Y]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[8][p] - q_[p][Y][Z]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[9][p] - q_[p][Z][Z]) < TEST_DOUBLE_TOLERANCE);
    */
  }

  info("ok\n");

  info("Checking normalisers norm_[i]*wv[p]*ma_[i][p]*ma_[j][p] = dij... ");

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
  info("ok\n");

  info("Checking ma_[i][p]*mi_[p][j] = dij ... ");

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
  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_model_distributions
 *
 *  Test the distribution interface.
 *
 *****************************************************************************/

static void test_model_distributions(void) {

  int i, n, p;
  int index;
  int ndist;
  double fvalue, fvalue_expected;
  double u[NDIM];

  /* Tests of the basic distribution functions. */

  info("\n\n\nDistribution functions:\n");
  info("Allocate lattice sites...\n");

  init_site();
  index = 0;

  info("Allocated 1 site\n");

  /* Report the number of distributions */

  ndist = distribution_ndist();

  info("Number of distributions: %d\n", ndist);
  test_assert(ndist == 1);

  for (n = 0; n < ndist; n++) {

    info("\n");
    info("Distribution %2d\n\n", n);

    info("Check individual distributions... ");

    for (p = 0; p < NVEL; p++) {
      fvalue_expected = 1.0*n + wv[p];
      distribution_f_set(index, p, n, fvalue_expected);
      fvalue = distribution_f(index, p, n);
      test_assert(fabs(fvalue - fvalue_expected) < TEST_DOUBLE_TOLERANCE);
    }
    info("ok\n");

    info("Check zeroth moment... ");

    fvalue_expected = 1.0*n*NVEL + 1.0;
    fvalue = distribution_zeroth_moment(index, n);
    test_assert(fabs(fvalue - fvalue_expected) < TEST_DOUBLE_TOLERANCE);
    info("ok\n");

    info("Check first moment... ");

    distribution_first_moment(index, n, u);
    for (i = 0; i < NDIM; i++) {
      test_assert(fabs(u[i] - 0.0) < TEST_DOUBLE_TOLERANCE);
    }
    info("ok\n");
  }

  finish_site();

  return;
}

/*****************************************************************************
 *
 *  test_model_halo_swap
 *
 *  Test full halo swap.
 *
 *****************************************************************************/

static void test_model_halo_swap() {

  int i, j, k, p;
  int n, ndist;
  int index, nlocal[3];
  const int nextra = 1;  /* Distribution halo width always 1 */
  double f_expect;
  double f_actual;

  info("\nHalo swap (full distributions)...\n\n");

  init_site();
  distribution_halo_set_complete();
  get_N_local(nlocal);
  ndist = distribution_ndist();

  /* The test relies on a uniform decomposition in parallel:
   *
   * f[0] or f[X] is set to local x index,
   * f[1] or f[Y] is set to local y index
   * f[2] or f[Z] is set to local z index
   * remainder are set to velocity index. */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {

	index = get_site_index(i, j, k);

	for (n = 0; n < ndist; n++) {
	  distribution_f_set(index, X, n, (double) (i));
	  distribution_f_set(index, Y, n, (double) (j));
	  distribution_f_set(index, Z, n, (double) (k));

	  for (p = 3; p < NVEL; p++) {
	    distribution_f_set(index, p, n, (double) p);
	  }
	}
      }
    }
  }

  halo_site();

  /* Test all the sites not in the interior */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = get_site_index(i, j, k);

	for (n = 0; n < ndist; n++) {

	  f_expect = fabs(i - nlocal[X]);
	  f_actual = distribution_f(index, X, n);
	  test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);

	  f_expect = fabs(j - nlocal[Y]);
	  f_actual = distribution_f(index, Y, n);
	  test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);

	  f_expect = fabs(k - nlocal[Z]);
	  f_actual = distribution_f(index, Z, n);
	  test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);

	  for (p = 3; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, n);
	    f_expect = (double) p;
	    test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	  }
	}
      }
    }
  }

  info("Halo swap ok\n");
  finish_site();
}

/*****************************************************************************
 *
 *  test_model_reduced_halo_swap
 *
 *****************************************************************************/

static void test_model_reduced_halo_swap() {  

  int i, j, k, p;
  int icdt, jcdt, kcdt;
  int index, nlocal[3];
  int n, ndist;
  const int nextra = 1;

  double f_expect;
  double f_actual;

  info("\nHalo swap (reduced)...\n\n");

  init_site();
  distribution_halo_set_reduced();
  get_N_local(nlocal);
  ndist = distribution_ndist();

  /* Set everything which is NOT in a halo */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = get_site_index(i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    f_expect = 1.0*(n*NVEL + p);
	    distribution_f_set(index, p, n, f_expect);
	  }
	}
      }
    }
  }

  halo_site();

  /* Now check that the interior sites are unchanged */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = get_site_index(i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, n);
	    f_expect = 1.0*(n*NVEL +  p);
	    test_assert(fabs(f_expect - f_actual) < TEST_DOUBLE_TOLERANCE);
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

	index = get_site_index(i, j, k);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < NVEL; p++) {

	    f_actual = distribution_f(index, p, n);
	    f_expect = 1.0*(n*NVEL + p);

	    icdt = i + cv[p][X];
	    jcdt = j + cv[p][Y];
	    kcdt = k + cv[p][Z];

	    if (test_model_is_domain(icdt, jcdt, kcdt)) {
	      test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	    }
	  }
	}

	/* Next site */
      }
    }
  }

  info("Reduced halo swapping... ok");

  return;
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

  get_N_local(nlocal);

  if (ic < 1) iam = 0;
  if (jc < 1) iam = 0;
  if (kc < 1) iam = 0;
  if (ic > nlocal[X]) iam = 0;
  if (jc > nlocal[Y]) iam = 0;
  if (kc > nlocal[Z]) iam = 0;

  return iam;
}
