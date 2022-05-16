/*****************************************************************************
 *
 *  test_pair_ss_cut_ij.c
 *
 *  Edinburgh Soft Matter and Statictical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "pair_ss_cut_ij.h"
#include "tests.h"

int test_pair_ss_cut_ij_create(pe_t * pe, cs_t * cs);
int test_pair_ss_cut_ij_param_set(pe_t * pe, cs_t * cs);
int test_pair_ss_cut_ij_single(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_pair_ss_cut_ij_suite
 *
 *****************************************************************************/

int test_pair_ss_cut_ij_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_pair_ss_cut_ij_create(pe, cs);
  test_pair_ss_cut_ij_param_set(pe, cs);
  test_pair_ss_cut_ij_single(pe, cs);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_pair_ss_cut_ij\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_ss_cut_ij_create
 *
 *****************************************************************************/

int test_pair_ss_cut_ij_create(pe_t * pe, cs_t * cs) {

  pair_ss_cut_ij_t * obj = NULL;

  double epsilon[2] = {0};
  double sigma[2] = {0};
  double nu[2] = {0};
  double hc[2] = {0};

  pair_ss_cut_ij_create(pe, cs, 2, epsilon, sigma, nu, hc, &obj);
  assert(obj);

  assert(obj->ntypes == 2);
  assert(obj->epsilon);
  assert(obj->sigma);
  assert(obj->nu);
  assert(obj->hc);

  for (int i = 0; i < obj->ntypes; i++) {
    for (int j = 0; j < obj->ntypes; j++) {
      assert(fabs(obj->epsilon[i][j] - 0.0) < DBL_EPSILON);
      assert(fabs(obj->sigma[i][j]   - 0.0) < DBL_EPSILON);
      assert(fabs(obj->nu[i][j]      - 0.0) < DBL_EPSILON);
      assert(fabs(obj->hc[i][j]      - 0.0) < DBL_EPSILON);
    }
  }

  pair_ss_cut_ij_free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_ss_cut_ij_param_set
 *
 *****************************************************************************/

int test_pair_ss_cut_ij_param_set(pe_t * pe, cs_t * cs) {

  pair_ss_cut_ij_t * obj = NULL;

  double epsilon[3] = {1.0, 2.0, 3.0}; /* 11 12 22 */
  double sigma[3]   = {4.0, 5.0, 6.0};
  double nu[3]      = {0.5, 1.5, 2.5};
  double hc[3]      = {7.0, 8.0, 9.0};

  assert(pe);
  assert(cs);

  pair_ss_cut_ij_create(pe, cs, 2, epsilon, sigma, nu, hc, &obj);

  {
    pair_ss_cut_ij_param_set(obj, epsilon, sigma, nu, hc);

    /* I'm going to write this out explicitly ... */
    assert(fabs(obj->epsilon[0][0] - epsilon[0]) < DBL_EPSILON);
    assert(fabs(obj->epsilon[0][1] - epsilon[1]) < DBL_EPSILON);
    assert(fabs(obj->epsilon[1][0] - epsilon[1]) < DBL_EPSILON);
    assert(fabs(obj->epsilon[1][1] - epsilon[2]) < DBL_EPSILON);

    assert(fabs(obj->sigma[0][0]   - sigma[0]  ) < DBL_EPSILON);
    assert(fabs(obj->sigma[0][1]   - sigma[1]  ) < DBL_EPSILON);
    assert(fabs(obj->sigma[1][0]   - sigma[1]  ) < DBL_EPSILON);
    assert(fabs(obj->sigma[1][1]   - sigma[2]  ) < DBL_EPSILON);

    assert(fabs(obj->nu[0][0]      - nu[0]     ) < DBL_EPSILON);
    assert(fabs(obj->nu[0][1]      - nu[1]     ) < DBL_EPSILON);
    assert(fabs(obj->nu[1][0]      - nu[1]     ) < DBL_EPSILON);
    assert(fabs(obj->nu[1][1]      - nu[2]     ) < DBL_EPSILON);

    assert(fabs(obj->hc[0][0]      - hc[0]     ) < DBL_EPSILON);
    assert(fabs(obj->hc[0][1]      - hc[1]     ) < DBL_EPSILON);
    assert(fabs(obj->hc[1][0]      - hc[1]     ) < DBL_EPSILON);
    assert(fabs(obj->hc[1][1]      - hc[2]     ) < DBL_EPSILON);
  }

  pair_ss_cut_ij_free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_ss_cut_ij_single
 *
 *****************************************************************************/

int test_pair_ss_cut_ij_single(pe_t * pe, cs_t * cs) {

  pair_ss_cut_ij_t * obj = NULL;

  double epsilon[3] = {1.0, 0.0, 0.0}; /* 00 interactions only */
  double sigma[3]   = {1.0, 1.0, 1.0};
  double nu[3]      = {1.0, 1.0, 1.0};
  double hc[3]      = {2.0, 2.0, 2.0};

  assert(pe);
  assert(cs);

  pair_ss_cut_ij_create(pe, cs, 2, epsilon, sigma, nu, hc, &obj);

  {
    double h = 1.0;
    double f = 0.0;
    double v = 0.0;
    pair_ss_cut_ij_single(obj, 0, 0, h, &v, &f);
    assert(fabs(f - 0.25) < DBL_EPSILON);
    assert(fabs(v - 0.75) < DBL_EPSILON);

    pair_ss_cut_ij_single(obj, 0, 1, h, &v, &f);
    assert(fabs(f - 0.00) < DBL_EPSILON);
    assert(fabs(v - 0.00) < DBL_EPSILON);

    pair_ss_cut_ij_single(obj, 1, 0, h, &v, &f);
    assert(fabs(f - 0.00) < DBL_EPSILON);
    assert(fabs(v - 0.00) < DBL_EPSILON);

    pair_ss_cut_ij_single(obj, 1, 1, h, &v, &f);
    assert(fabs(f - 0.00) < DBL_EPSILON);
    assert(fabs(v - 0.00) < DBL_EPSILON);
  }

  pair_ss_cut_ij_free(obj);

  return 0;
}
