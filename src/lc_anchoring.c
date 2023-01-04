/*****************************************************************************
 *
 *  lc_anchoring.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "lc_anchoring.h"
#include "util.h"

/*****************************************************************************
 *
 *  lc_anchoring_type_from_string
 *
 *  Translate a string to an anchoring type
 *
 ****************************************************************************/

lc_anchoring_enum_t lc_anchoring_type_from_string(const char * string) {

  lc_anchoring_enum_t lc_anchor = LC_ANCHORING_INVALID;

  assert(string);

  if (strcmp(string, "normal") == 0) lc_anchor = LC_ANCHORING_NORMAL;
  if (strcmp(string, "planar") == 0) lc_anchor = LC_ANCHORING_PLANAR;
  if (strcmp(string, "fixed")  == 0) lc_anchor = LC_ANCHORING_FIXED;

  return lc_anchor;
}

/*****************************************************************************
 *
 *  lc_anchoring_type_from_enum
 *
 *  Return the relevant descriptive string.
 *
 *****************************************************************************/

const char * lc_anchoring_type_from_enum(lc_anchoring_enum_t type) {

  switch (type) {
  case LC_ANCHORING_NORMAL:
    return "normal";
    break;
  case LC_ANCHORING_PLANAR:
    return "planar";
    break;
  case LC_ANCHORING_FIXED:
    return "fixed";
    break;
  case LC_ANCHORING_INVALID:
    return "invalid";
    break;
  default:
    /* Nothing */
    break;
  }

  return "invalid";
}

/*****************************************************************************
 *
 *  lc_anchoring_matrix1
 *
 *  Coefficients in the gradient terms.
 *
 *  There are 3 cases (flat face in each coordinate direction) each with
 *  six equations; for each case we have a diagonal matrix for which the
 *  inverse may be computed in simple fashion...
 *
 *  We only store the diagonal elements a[3][6].
 *
 *****************************************************************************/

int lc_anchoring_matrix1(double kappa0, double kappa1, double a[3][6]) {

  /* Normals in six-point stencil */
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

  for (int ia = 0; ia < 3; ia++) {

    double bc[6][6][3] = {0};

    lc_anchoring_coefficients(kappa0, kappa1, bcs[2*ia + 1], bc);

    for (int n1 = 0; n1 < 6; n1++) {
      a[ia][n1] = 1.0/bc[n1][n1][ia];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lc_anchoring_matrix12
 *
 *  Three different cases here for "edges" - the join between two
 *  adjacent faces. The relevant normals can be x and y, x and z,
 *  or y and z.
 *  
 *
 *****************************************************************************/

int lc_anchoring_matrix2(double kappa0, double kappa1, double a[3][12][12]) {

  int ifail = 0;

  double a18[18][18] = {0};
  double ** a12inv[3] = {0};

  /* Compute inverse matrices */

  util_matrix_create(12, 12, &(a12inv[0]));
  util_matrix_create(12, 12, &(a12inv[1]));
  util_matrix_create(12, 12, &(a12inv[2]));

  for (int ia = 0; ia < 3; ia++) {

    int n[3] = {0}; n[ia] = 1; /* Unit normal */
    double bc[6][6][3] = {0};

    lc_anchoring_coefficients(kappa0, kappa1, n, bc);

    for (int n1 = 0; n1 < NSYMM; n1++) {
      for (int n2 = 0; n2 < NSYMM; n2++) {
	for (int ib = 0; ib < 3; ib++) {
	  double dab = (ia == ib);
	  a18[ia*NSYMM + n1][ib*NSYMM + n2] = 0.5*(1.0 + dab)*bc[n1][n2][ib];
	}
      }
    }
  }

  /* xy: just the first 12 rows ... */
  for (int n1 = 0; n1 < 12; n1++) {
    for (int n2 = 0; n2 < 12; n2++) {
      a12inv[0][n1][n2] = a18[n1][n2];
    }
  }

  /* xz: first 6 and last six rows */
  for (int n1 = 0; n1 < 6; n1++) {
    for (int n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][  n2] = a18[n1][n2];
      a12inv[1][n1][6+n2] = a18[n1][12+n2];
    }
  }

  for (int n1 = 6; n1 < 12; n1++) {
    for (int n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][  n2] = a18[6+n1][n2];
      a12inv[1][n1][6+n2] = a18[6+n1][12+n2];
    }
  }

  /* yz: last twelve rows */
  for (int n1 = 0; n1 < 12; n1++) {
    for (int n2 = 0; n2 < 12; n2++) {
      a12inv[2][n1][n2] = a18[6+n1][6+n2];
    }
  }

  ifail += util_matrix_invert(12, a12inv[0]);
  ifail += util_matrix_invert(12, a12inv[1]);
  ifail += util_matrix_invert(12, a12inv[2]);

  for (int ia = 0; ia < 3; ia++) {
    for (int n1 = 0; n1 < 12; n1++) {
      for (int n2 = 0; n2 < 12; n2++) {
	a[ia][n1][n2] = a12inv[ia][n1][n2];
      }
    }
  }

  util_matrix_free(12, &(a12inv[2]));
  util_matrix_free(12, &(a12inv[1]));
  util_matrix_free(12, &(a12inv[0]));

  return ifail;
}

/*****************************************************************************
 *
 *  lc_anchoring_matrix3
 *
 *  The most general case we consider: three faces meeting at a corner.
 *  This gives rise to a single 18x18 matrix.
 *
 *****************************************************************************/

int lc_anchoring_matrix3(double kappa0, double kappa1, double a18[18][18]) {

  int ifail = 0;

  /* Compute system matrix for utility routine */
  double ** a18inv = {0};

  util_matrix_create(18, 18, &a18inv);

  for (int ia = 0; ia < 3; ia++) {

    int n[3] = {0}; n[ia] = 1; /* Unit normal */
    double bc[6][6][3] = {0};

    lc_anchoring_coefficients(kappa0, kappa1, n, bc);

    /* One 6x18 block */
    for (int n1 = 0; n1 < NSYMM; n1++) {
      for (int n2 = 0; n2 < NSYMM; n2++) {
	/* The factor restores the factor 1/2 in off-diagonal blocks
	 * cf the raw bc[][][] values */
	for (int ib = 0; ib < 3; ib++) {
	  double dab = (ia == ib);
	  a18inv[ia*NSYMM + n1][ib*NSYMM + n2] = 0.5*(1.0+dab)*bc[n1][n2][ib];
	}
      }
    }
  }

  /* And invert ... */

  ifail = util_matrix_invert(18, a18inv);

  for (int n1 = 0; n1 < 18; n1++) {
    for (int n2 = 0; n2 < 18; n2++) {
      a18[n1][n2] = a18inv[n1][n2];
    }
  }

  util_matrix_free(18, &a18inv);

  return ifail;
}

/*****************************************************************************
 *
 *  lc_anchoring_matrices
 *
 *****************************************************************************/

int lc_anchoring_matrices(double kappa0, double kappa1,
			  lc_anchoring_matrices_t * matrices) {

  int ifail = 0;

  assert(matrices);

  ifail += lc_anchoring_matrix1(kappa0, kappa1, matrices->a6inv);
  ifail += lc_anchoring_matrix2(kappa0, kappa1, matrices->a12inv);
  ifail += lc_anchoring_matrix3(kappa0, kappa1, matrices->a18inv);

  return ifail;
}
