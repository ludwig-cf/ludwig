/*****************************************************************************
 *
 *  test_model.c
 *
 *  Tests for model data: distributions, halos, i/o (pending!).
 *  PENDING: This is to be merged with test_halo.c under "test_lb_data.c".
 *  PENDING: Coverage check.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
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
#include "lb_data.h"
#include "tests.h"

static void test_model_velocity_set(void);

int do_test_model_distributions(pe_t * pe, cs_t * cs);
int do_test_model_halo_swap(pe_t * pe, cs_t * cs);
int do_test_model_reduced_halo_swap(pe_t * pe, cs_t * cs);
int do_test_lb_model_io(pe_t * pe, cs_t * cs);
static  int test_model_is_domain(cs_t * cs, int ic, int jc, int kc);


/* Utility to return a unique value for global (ic,jc,kc,p) */
/* This allows e.g., tests to check distribution values in parallel
 * exchanges. */

/* (ic, jc, kc) are local indices */
/* Result could be unsigned integer... */

#include <stdint.h>

int64_t lb_data_index(lb_t * lb, int ic, int jc, int kc, int p) {

  int64_t index = INT64_MIN;
  int64_t nall[3] = {};
  int64_t nstr[3] = {};
  int64_t pstr    = 0;

  int ntotal[3] = {};
  int offset[3] = {};
  int nhalo = 0;

  assert(lb);
  assert(0 <= p && p < lb->model.nvel);

  cs_ntotal(lb->cs, ntotal);
  cs_nlocal_offset(lb->cs, offset);
  cs_nhalo(lb->cs, &nhalo);

  nall[X] = ntotal[X] + 2*nhalo;
  nall[Y] = ntotal[Y] + 2*nhalo;
  nall[Z] = ntotal[Z] + 2*nhalo;
  nstr[Z] = 1;
  nstr[Y] = nstr[Z]*nall[Z];
  nstr[X] = nstr[Y]*nall[Y];
  pstr    = nstr[X]*nall[X];

  {
    int igl = offset[X] + ic;
    int jgl = offset[Y] + jc;
    int kgl = offset[Z] + kc;

    /* A periodic system */
    igl = igl % ntotal[X];
    jgl = jgl % ntotal[Y];
    kgl = kgl % ntotal[Z];
    if (igl < 1) igl = igl + ntotal[X];
    if (jgl < 1) jgl = jgl + ntotal[Y];
    if (kgl < 1) kgl = kgl + ntotal[Z];

    assert(1 <= igl && igl <= ntotal[X]);
    assert(1 <= jgl && jgl <= ntotal[Y]);
    assert(1 <= kgl && kgl <= ntotal[Z]);

    index = pstr*p + nstr[X]*igl + nstr[Y]*jgl + nstr[Z]*kgl;
  }

  return index;
}

/*****************************************************************************
 *
 *  util_lb_data_check_set
 *
 *  Set unique test values in the distribution.
 * 
 *****************************************************************************/

int util_lb_data_check_set(lb_t * lb) {

  int nlocal[3] = {};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	for (int p = 0 ; p < lb->model.nvel; p++) {
	  int index = cs_index(lb->cs, ic, jc, kc);
	  int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	  lb->f[laddr] = 1.0*lb_data_index(lb, ic, jc, kc, p); 
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_lb_data_check
 *
 *  Examine halo values and check they are as expected.
 *
 *****************************************************************************/

int util_lb_data_check(lb_t * lb, int full) {

  int ifail = 0;
  int nh = 1;
  int nhk = nh;
  int nlocal[3] = {};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  /* Fix for 2d, where there should be no halo regions in Z */
  if (lb->ndim == 2) nhk = 0;

  for (int ic = 1 - nh; ic <= nlocal[X] + nh; ic++) {
    for (int jc = 1 - nh; jc <= nlocal[Y] + nh; jc++) {
      for (int kc = 1 - nhk; kc <= nlocal[Z] + nhk; kc++) {

	int is_halo = (ic < 1 || jc < 1 || kc < 1 ||
		       ic > nlocal[X] || jc > nlocal[Y] || kc > nlocal[Z]);

	if (is_halo == 0) continue;

	int index = cs_index(lb->cs, ic, jc, kc);

	for (int p = 0; p < lb->model.nvel; p++) {

	  /* Look for propagating distributions (into domain). */
	  int icdt = ic + lb->model.cv[p][X];
	  int jcdt = jc + lb->model.cv[p][Y];
	  int kcdt = kc + lb->model.cv[p][Z];

	  is_halo = (icdt < 1 || jcdt < 1 || kcdt < 1 ||
		     icdt > nlocal[X] || jcdt > nlocal[Y] || kcdt > nlocal[Z]);

	  if (full || is_halo == 0) {
	    /* Check */
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	    double fex = 1.0*lb_data_index(lb, ic, jc, kc, p);
	    if (fabs(fex - lb->f[laddr]) > DBL_EPSILON) ifail += 1;
	    assert(fabs(fex - lb->f[laddr]) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_halo_post_wait
 *
 *****************************************************************************/

int test_lb_halo_post_wait(pe_t * pe, cs_t * cs, int ndim, int nvel, int full) {

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  options.ndim = ndim;
  options.nvel = nvel;
  lb_data_create(pe, cs, &options, &lb);

  util_lb_data_check_set(lb);

  {
    lb_halo_t h = {};
    lb_halo_create(lb, &h, LB_HALO_OPENMP_FULL);
    lb_halo_post(lb, &h);
    lb_halo_wait(lb, &h);
    lb_halo_free(lb, &h);
  }

  util_lb_data_check(lb, full);
  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_halo
 *
 *****************************************************************************/

int test_lb_halo(pe_t * pe) {

  assert(pe);

  /* Two dimensional system */
  {
    cs_t * cs = NULL;
    int ntotal[3] = {64, 64, 1};

    cs_create(pe, &cs);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);

    test_lb_halo_post_wait(pe, cs, 2, 9, LB_HALO_OPENMP_REDUCED);
    test_lb_halo_post_wait(pe, cs, 2, 9, LB_HALO_OPENMP_FULL);

    cs_free(cs);
  }

  /* Three dimensional system */
  {
    cs_t * cs = NULL;

    cs_create(pe, &cs);
    cs_init(cs);

    test_lb_halo_post_wait(pe, cs, 3, 15, LB_HALO_OPENMP_REDUCED);
    test_lb_halo_post_wait(pe, cs, 3, 15, LB_HALO_OPENMP_FULL);
    test_lb_halo_post_wait(pe, cs, 3, 19, LB_HALO_OPENMP_REDUCED);
    test_lb_halo_post_wait(pe, cs, 3, 19, LB_HALO_OPENMP_FULL);
    test_lb_halo_post_wait(pe, cs, 3, 27, LB_HALO_OPENMP_REDUCED);
    test_lb_halo_post_wait(pe, cs, 3, 27, LB_HALO_OPENMP_FULL);

    cs_free(cs);
  }

  return 0;
}


/*****************************************************************************
 *
 *  test_model_suite
 *
 *****************************************************************************/

int test_model_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_halo(pe);

  cs_create(pe, &cs);
  cs_init(cs);

  /* Test model structure (coordinate-independent stuff) */

  test_model_velocity_set();

  /* Now test actual distributions */

  do_test_model_distributions(pe, cs);
  do_test_model_halo_swap(pe, cs);
  do_test_model_reduced_halo_swap(pe, cs);
  do_test_lb_model_io(pe, cs);

  pe_info(pe, "PASS     ./unit/test_model\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_model_velocity_set
 *
 *  Some residual older tests which remain relevant.
 *
 *****************************************************************************/

static void test_model_velocity_set(void) {

  test_assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  printf("Compiled model NDIM %2d NVEL %2d\n", NDIM, NVEL);
  printf("sizeof(lb_collide_param_t) %ld bytes\n", sizeof(lb_collide_param_t));

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

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  /* Tests of the basic distribution functions. */

  options.ndim  = NDIM;
  options.nvel  = NVEL;   
  options.ndist = ndist;

  lb_data_create(pe, cs, &options, &lb);
  assert(lb);
  assert(lb->ndist == ndist);

  for (n = 0; n < ndist; n++) {
    for (p = 0; p < lb->model.nvel; p++) {
      fvalue_expected = 0.01*n + lb->model.wv[p];
      lb_f_set(lb, index, p, n, fvalue_expected);
      lb_f(lb, index, p, n, &fvalue);
      assert(fabs(fvalue - fvalue_expected) < DBL_EPSILON);
    }

    /* Check zeroth moment... */

    fvalue_expected = 0.01*n*lb->model.nvel + 1.0;
    lb_0th_moment(lb, index, (lb_dist_enum_t) n, &fvalue);
    assert(fabs(fvalue - fvalue_expected) <= DBL_EPSILON);

    /* Check first moment... */

    lb_1st_moment(lb, index, (n == 0) ? LB_RHO : LB_PHI, u);

    for (i = 0; i < lb->model.ndim; i++) {
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

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  options.ndim = NDIM;
  options.nvel = NVEL;
  options.ndist = ndist;
  lb_data_create(pe, cs, &options, &lb);

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

	  for (p = 3; p < lb->model.nvel; p++) {
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

	  for (p = 3; p < lb->model.nvel; p++) {
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
  int n, ndist = 1;
  const int nextra = 1;

  double f_expect;
  double f_actual;

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  options.ndim = NDIM;
  options.nvel = NVEL;
  options.ndist = ndist;
  options.halo = LB_HALO_OPENMP_REDUCED;
  lb_data_create(pe, cs, &options, &lb);
  assert(lb);

  cs_nlocal(cs, nlocal);

  /* Set everything which is NOT in a halo */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    f_expect = 1.0*(n*lb->model.nvel + p);
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
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*lb->model.nvel +  p);
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
	  for (p = 0; p < lb->model.nvel; p++) {

	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*lb->model.nvel + p);

	    icdt = i + lb->model.cv[p][X];
	    jcdt = j + lb->model.cv[p][Y];
	    kcdt = k + lb->model.cv[p][Z];

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

  assert(pe);
  assert(cs);

  /* Write */
  /* Read */

  /* Compare */

  return 0;
}
