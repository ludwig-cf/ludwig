/*****************************************************************************
 *
 *  ut_util.c
 *
 *  Unit test for utilities.
 *
 *****************************************************************************/

#include <assert.h>
#include "unit_control.h"
#include "util.h"

int do_test_util_constants(control_t * ctrl);
int do_test_util_dij(control_t * ctrl);
int do_test_util_eijk(control_t * ctrl);
int do_test_util_discrete_volume_sphere(control_t * ctrl);
int do_test_util_matrix(control_t * ctrl);
int do_test_util_svd(control_t * ctrl);
int do_test_util_ran_uniform(control_t * ctrl);
int do_test_util_ran_gaussian(control_t * ctrl);
int do_test_util_ran_unit_vector(control_t * ctrl);

/* Statistical sample size for RNG tests; the tolerance STAT_EPSILON
 * needs to be about 1/sqrt(STAT_NSAMPLE) */

#define STAT_NSAMPLE 10000000
#define STAT_EPSILON 0.001

/*****************************************************************************
 *
 *  do_ut_util
 *
 *****************************************************************************/

int do_ut_util(control_t * ctrl) {

  do_test_util_constants(ctrl);
  do_test_util_dij(ctrl);
  do_test_util_eijk(ctrl);
  do_test_util_discrete_volume_sphere(ctrl);
  do_test_util_matrix(ctrl);
  do_test_util_svd(ctrl);
  do_test_util_ran_uniform(ctrl);
  do_test_util_ran_gaussian(ctrl);
  do_test_util_ran_unit_vector(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_constants
 *
 *****************************************************************************/

int do_test_util_constants(control_t * ctrl) {

  double pi;

  control_test(ctrl, __CONTROL_INFO__);

  try {
    pi = 4.0*atan(1.0);
    control_verb(ctrl, "pi_ is %22.15e (%22.15e)\n", pi, pi_);
    control_macro_test_dbl_eq(ctrl, pi_, pi, DBL_EPSILON);

    control_verb(ctrl, "r3_ is 1/3 (%22.15e)\n", r3_);
    control_macro_test_dbl_eq(ctrl, r3_, (1.0/3.0), DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_dij
 *
 *****************************************************************************/

int do_test_util_dij(control_t * ctrl) {

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Kronecker delta dij\n");

  try {

    control_macro_test_dbl_eq(ctrl, d_[0][0], 1.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, d_[1][1], 1.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, d_[2][2], 1.0, DBL_EPSILON);

    control_macro_test_dbl_eq(ctrl, d_[0][1], 0.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, d_[0][2], 0.0, DBL_EPSILON);

    control_macro_test_dbl_eq(ctrl, d_[1][0], 0.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, d_[1][2], 0.0, DBL_EPSILON);

    control_macro_test_dbl_eq(ctrl, d_[2][0], 0.0, DBL_EPSILON);
    control_macro_test_dbl_eq(ctrl, d_[2][1], 0.0, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_eijk
 *
 *  Permutations
 *
 *****************************************************************************/

int do_test_util_eijk(control_t * ctrl) {

  int i, j, k;
  int m, n;
  double sume, sumd;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Permutation tensor\n");

  try {
    /* Permutations */
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
	for (k = 0; k < 3; k++) {
	  sume = e_[i][j][k] + e_[i][k][j];
	  control_macro_test_dbl_eq(ctrl, sume, 0.0, DBL_EPSILON);
	  sume = e_[i][j][k] + e_[j][i][k];
	  control_macro_test_dbl_eq(ctrl, sume, 0.0, DBL_EPSILON);
	  sume = e_[i][j][k] - e_[k][i][j];
	  control_macro_test_dbl_eq(ctrl, sume, 0.0, DBL_EPSILON);
	}
      }
    }

    /* The identity e_ijk e_imn = d_jm d_kn - d_jn d_km */

    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (m = 0; m < 3; m++) {
	  for (n = 0; n < 3; n++) {
	    sume = 0.0;
	    for (i = 0; i < 3; i++) {
	      sume += e_[i][j][k]*e_[i][m][n];
	    }
	    sumd = d_[j][m]*d_[k][n] - d_[j][n]*d_[k][m];
	    control_macro_test_dbl_eq(ctrl, sume, sumd, DBL_EPSILON);
	  }
	}
      }
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_discrete_volume_sphere
 *
 *  Test values from separate code to compute varience as a function
 *  of discrete colloid size as function of radius on lattice.
 *
 *****************************************************************************/

int do_test_util_discrete_volume_sphere(control_t * ctrl) {

  int unit_assert_dv(control_t * ctrl, double rx, double ry, double rz,
		     double a, double v);

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Position radius volume (expected)\n");

  try {
    unit_assert_dv(ctrl, 0.00, 0.00, 0.00, 1.00, 1.00);
    unit_assert_dv(ctrl, 0.50, 0.50, 0.50, 1.00, 8.00);

    unit_assert_dv(ctrl, 0.52, 0.10, 0.99, 1.25, 10.0);
    unit_assert_dv(ctrl, 0.52, 0.25, 0.99, 1.25,  8.0);

    unit_assert_dv(ctrl, 0.00, 0.00, 0.00, 2.30, 57.0);
    unit_assert_dv(ctrl, 0.50, 0.50, 0.50, 2.30, 56.0);

    unit_assert_dv(ctrl, 0.52, 0.10, 0.99, 4.77, 461.0);
    unit_assert_dv(ctrl, 0.52, 0.25, 0.99, 4.77, 453.0);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

int unit_assert_dv(control_t * ctrl, double rx, double ry, double rz,
		   double a, double vtrue)  throws (TestFailedException) {

  double r[3];
  double v;
  double dx = 100.0;
  double dy = 100.0;
  double dz = 100.0;

  assert(ctrl);

  r[0] = rx; r[1] = ry; r[2] = rz;
  util_discrete_volume_sphere(r, a, &v);
  control_verb(ctrl, "%10.3e %10.3e %10.3e %10.3e %10.3e (%10.3e)\n",
	       r[0], r[1], r[2], a, v, vtrue);
  control_macro_test_dbl_eq(ctrl, v, vtrue, DBL_EPSILON);

  /* +ve coordinates */

  r[0] = rx + dx; r[1] = ry + dy; r[2] = rz + dz;
  util_discrete_volume_sphere(r, a, &v);
  control_verb(ctrl, "%10.3e %10.3e %10.3e %10.3e %10.3e (%10.3e)\n",
	       r[0], r[1], r[2], a, v, vtrue);
  control_macro_test_dbl_eq(ctrl, v, vtrue, DBL_EPSILON);

  /* -ve coordinates */

  r[0] = rx - dx; r[1] = ry - dy; r[2] = rz - dz;
  util_discrete_volume_sphere(r, a, &v);
  control_verb(ctrl, "%10.3e %10.3e %10.3e %10.3e %10.3e (%10.3e)\n",
	       r[0], r[1], r[2], a, v, vtrue);
  control_macro_test_dbl_eq(ctrl, v, vtrue, DBL_EPSILON);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_matrix
 *
 *****************************************************************************/

int do_test_util_matrix(control_t * ctrl) {

  int ifail = 0;
  int m, n;
  double ** a = NULL;

  MPI_Comm comm;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Matrix object allocation\n");
  control_comm(ctrl, &comm);

  try {
    m = 2; n = 2;
    ifail = util_matrix_create(m, n, &a);
    control_macro_assert(ctrl, ifail == 0, NullPointerException);

    a[0][0] = 0.0; a[0][1] = 0.0;
    a[1][0] = 0.0; a[1][1] = 0.0;

    util_matrix_free(m, &a);
  }
  catch (NullPointerException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    /* just report a fail ... */
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_svd
 *
 *****************************************************************************/

int do_test_util_svd(control_t * ctrl) {

  int ifail = 0;
  int m = 3;
  int n = 2;
  double ** a = NULL;
  int unit_assert_svd_matrix(control_t * ctrl, int m, int n, double ** a);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Singular value decomposition routine\n");

  try {
    ifail = util_matrix_create(m, n, &a);
    control_macro_assert(ctrl, ifail == 0, NullPointerException);

    a[0][0] = -1.0;
    a[0][1] =  0.0;
    a[1][0] =  0.0;
    a[1][1] =  3.0;
    a[2][0] =  2.0;
    a[2][1] = -1.0;

    unit_assert_svd_matrix(ctrl, m, n, a);
  }
  catch (NullPointerException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    if (a) util_matrix_free(m, &a);
  }

  control_report(ctrl);

  return 0;
}


/*****************************************************************************
 *
 *  unit_assert_svd_matrix
 *
 *  Test SVD routine by mutliplying out decomposition and checking
 *  against original for a simple case.
 *
 *  The incoming m by n matrix a is preserved.
 *
 *****************************************************************************/

int unit_assert_svd_matrix(control_t * ctrl, int m, int n, double ** a)
  throws (TestFailedException) {

  int i, j, k;
  int ifail = 0;
  double sum;
  double svd_epsilon;     /* Test tolerance for this a matrix */

  double ** u = NULL;     /* m x n matrix u (copy of a on input to svd) */
  double * w = NULL;      /* Singular values w[n] */
  double ** v = NULL;     /* n x n matrix v */

  assert(ctrl);
  assert(a);

  try {

    ifail = util_matrix_create(m, n, &u);
    control_macro_assert(ctrl, ifail == 0, NullPointerException);
    ifail = util_matrix_create(n, n, &v);
    control_macro_assert(ctrl, ifail == 0, NullPointerException);
    ifail = util_vector_create(n, &w);
    control_macro_assert(ctrl, ifail == 0, NullPointerException);

    /* Copy input matrix. Use largest fabs(a[i][j]) to set a tolerance */
    /* The factor of 2 required to pass under normal circumstances. */
  
    sum = 0.0;

    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
	u[i][j] = a[i][j];
	sum = dmax(sum, fabs(a[i][j]));
      }
    }

    svd_epsilon = 2.0*sum*DBL_EPSILON;

    control_macro_test(ctrl, util_svd(m, n, u, w, v) == 0);

    /* Assert u is orthonormal \sum_{k=0}^{m-1} u_ki u_kj = delta_ij
     * for 0 <= i, j < n: */

    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
	sum = 0.0;
	for (k = 0; k < m; k++) {
	  sum += u[k][j]*u[k][i];
	}
	if (fabs(sum - 1.0*(i == j)) > svd_epsilon) ifail += 1;
	control_macro_test_dbl_eq(ctrl, sum, 1.0*(i == j), svd_epsilon);
      }
    }

    /* Assert v is orthonormal \sum_{k=0}^{n-1} v_ki v_kj = delta_ij
     * for <= 0 i, j, < n: */

    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
	sum = 0.0;
	for (k = 0; k < n; k++) {
	  sum += v[k][j]*v[k][i];
	}
	control_macro_test_dbl_eq(ctrl, sum, 1.0*(i == j), svd_epsilon);
      }
    }
    control_macro_test(ctrl, ifail == 0);

    /* Assert u w v^t = a, ie., the decomposition is correct. */

    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
	sum = 0.0;
	for (k = 0; k < n; k++) {
	  sum += u[i][k]*w[k]*v[j][k];
	}
	control_macro_test_dbl_eq(ctrl, sum, a[i][j], svd_epsilon);
      }
    }

  }
  catch (NullPointerException) {
  }
  finally {
    if (w) util_vector_free(&w);
    if (v) util_matrix_free(n, &v);
    if (u) util_matrix_free(m, &u);
  }

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_ran_uniform
 *
 * See util.c implementation for s -> (as + c) % m (with c = 0)
 *
 * For uniform distribution, first moment = (1/2) and raw second moment
 * = (1/3); 10,000,000 just enough to clear FLT_EPSILON for rmin, rmax.
 *
 *****************************************************************************/

int do_test_util_ran_uniform(control_t * ctrl) {

  const int lcga = 1389796;
  const int lcgm = 2147483647;

  int n;
  int rank;
  int state;
  double ran1;
  double rmin, rmax;
  double rmom1, rmom2;

  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Uniform RNG test\n");
  control_comm(ctrl,  &comm);

  MPI_Comm_rank(comm, &rank);
  state = 1 + rank;

  try {
    util_ranlcg_reap_uniform(&state, &ran1);

    control_verb(ctrl, "First state:    %d (%d)\n", state, lcga);
    control_macro_test(ctrl, state == (1+rank)*lcga);

    control_verb(ctrl, "First var:      %22.15e (%22.15e)\n",
		 ran1, lcga*(1.0/lcgm));
    control_macro_test_dbl_eq(ctrl, ran1, state*(1.0/lcgm), DBL_EPSILON);

    /* Run statistics */

    rmin = 1.0; rmax = 0.0; rmom1 = 0.0; rmom2 = 0.0;

    for (n = 0; n < STAT_NSAMPLE; n++) {
      util_ranlcg_reap_uniform(&state, &ran1);
      rmin = dmin(ran1, rmin);
      rmax = dmax(ran1, rmax);
      rmom1 += ran1;
      rmom2 += ran1*ran1;
    }
    rmom1 /= STAT_NSAMPLE;
    rmom2 /= STAT_NSAMPLE;

    control_verb(ctrl, "Min ran1:       %22.15e\n", rmin);
    control_verb(ctrl, "Max ran1:       %22.15e\n", rmax);
    control_verb(ctrl, "First moment:   %22.15e\n", rmom1);
    control_verb(ctrl, "Second moment:  %22.15e\n", rmom2);

    control_macro_test(ctrl, rmin >= 0.0);
    control_macro_test(ctrl, rmax <= 1.0);

    control_macro_test(ctrl, rmin < STAT_EPSILON);
    control_macro_test(ctrl, (1.0 - rmax) < STAT_EPSILON);
    control_macro_test(ctrl, fabs(rmom1 - 0.5) < STAT_EPSILON);
    control_macro_test(ctrl, fabs(rmom2 - (1.0/3.0)) < STAT_EPSILON);

  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_ran_gaussian
 *
 *****************************************************************************/

int do_test_util_ran_gaussian(control_t * ctrl) {

  int n;
  int rank;
  int state ;
  double ran[2];
  double rmin, rmax;
  double rmom1, rmom2;

  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Gaussian RNG test\n");
  control_comm(ctrl,  &comm);

  MPI_Comm_rank(comm, &rank);
  state = 1 + rank;

  try {

    /* Run statistics */

    rmin = FLT_MAX; rmax = FLT_MIN; rmom1 = 0.0; rmom2 = 0.0;

    for (n = 0; n < STAT_NSAMPLE; n++) {
      util_ranlcg_reap_gaussian(&state, ran);
      rmin = dmin(ran[0], rmin);
      rmin = dmin(ran[1], rmin);
      rmax = dmax(ran[0], rmax);
      rmax = dmax(ran[1], rmax);
      rmom1 += ran[0] + ran[1];
      rmom2 += ran[0]*ran[0] + ran[1]*ran[1];
    }
    rmom1 /= 2*STAT_NSAMPLE;
    rmom2 /= 2*STAT_NSAMPLE;

    control_verb(ctrl, "Min ran1:       %22.15e\n", rmin);
    control_verb(ctrl, "Max ran1:       %22.15e\n", rmax);
    control_verb(ctrl, "First moment:   %22.15e\n", rmom1);
    control_verb(ctrl, "Second moment:  %22.15e\n", rmom2);

    control_macro_test(ctrl, rmin < -2.0);
    control_macro_test(ctrl, rmax > +2.0);
    control_macro_test(ctrl, fabs(rmom1) < STAT_EPSILON);
    control_macro_test(ctrl, fabs(rmom2 - 1.0) < STAT_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_util_ran_unit_vector
 *
 *****************************************************************************/

int do_test_util_ran_unit_vector(control_t * ctrl) {

  int n;
  int rank;
  int state;
  double rmin, rmax;
  double rmod;
  double runit[3];
  double rmean[3];

  MPI_Comm comm;

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Random unit vector test\n");
  control_comm(ctrl,  &comm);

  MPI_Comm_rank(comm, &rank);
  state = 1 + rank;

  try {

    rmin = 0.0; rmax = 0.0;
    rmean[0] = 0.0; rmean[1] = 0.0; rmean[2] = 0.0;

    for (n = 0; n < STAT_NSAMPLE; n++) {
      util_ranlcg_reap_unit_vector(&state, runit);
      rmean[0] += runit[0];
      rmean[1] += runit[1];
      rmean[2] += runit[2];
      rmin = dmin(runit[0], rmin);
      rmin = dmin(runit[1], rmin);
      rmin = dmin(runit[2], rmin);
      rmax = dmax(runit[0], rmax);
      rmax = dmax(runit[1], rmax);
      rmax = dmax(runit[2], rmax);

      /* Maximum discrepancy in modulus is about 4xDBL_EPSILON, but use
       * FLT_EPSILON and assume any problem will be gross. */

      rmod = runit[0]*runit[0] + runit[1]*runit[1] + runit[2]*runit[2];
      control_macro_test_dbl_eq(ctrl, rmod, 1.0, FLT_EPSILON);
    }

    rmean[0] /= STAT_NSAMPLE;
    rmean[1] /= STAT_NSAMPLE;
    rmean[2] /= STAT_NSAMPLE;

    control_verb(ctrl, "Component rmin %22.15e\n", rmin);
    control_verb(ctrl, "Component rmax %22.15e\n", rmax);
    control_macro_test_dbl_eq(ctrl, rmin, -1.0, STAT_EPSILON);
    control_macro_test_dbl_eq(ctrl, rmax, +1.0, STAT_EPSILON);

    control_verb(ctrl, "Component mean x: %22.15e\n", rmean[0]);
    control_verb(ctrl, "Component mean y: %22.15e\n", rmean[1]);
    control_verb(ctrl, "Component mean z: %22.15e\n", rmean[2]);
    control_macro_test_dbl_eq(ctrl, rmean[0], 0.0, STAT_EPSILON);
    control_macro_test_dbl_eq(ctrl, rmean[1], 0.0, STAT_EPSILON);
    control_macro_test_dbl_eq(ctrl, rmean[2], 0.0, STAT_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }

  control_report(ctrl);

  return 0;
}
