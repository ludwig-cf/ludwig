/*****************************************************************************
 *
 *  test_psi_sor.c
 *
 *  TODO: add tests via psi_stats_local()
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "psi_s.h"
#include "psi_sor.h"

#include "psi_stats.h"

static int do_test_sor1(void);
static int test_charge1_set(psi_t * psi);
static int test_charge1_exact(psi_t * obj, double tolerance);

int util_gauss_jordan(const int n, double * a, double * b);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  do_test_sor1();

  pe_finalise();
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_sor1
 *
 *  Set rho(z = 1)  = + (1/2NxNy)
 *      rho(z = Lz) = + (1/2NxNy)
 *      rho         = - 1/(NxNy*(Nz-2)) everywhere else.
 *
 *  This is a fully periodic system with zero total charge.
 *
 *****************************************************************************/

static int do_test_sor1(void) {

  double tol_abs = 0.01*FLT_EPSILON;
  double tol_rel = 1.00*FLT_EPSILON;

  /* Why am I using psi_ ? */

  coords_nhalo_set(1);
  coords_init();
  psi_create(2, &psi_);
  psi_valency_set(psi_, 0, +1.0);
  psi_valency_set(psi_, 1, -1.0);

  test_charge1_set(psi_);
  psi_halo_psi(psi_);
  psi_halo_rho(psi_);
  psi_sor_poisson(psi_, tol_abs, tol_rel);
  psi_sor_poisson(psi_, tol_abs, tol_rel);

  if (cart_size(Z) == 1) test_charge1_exact(psi_, tol_abs);

  psi_free(psi_);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_charge1_set
 *
 *  There is no sign, just a density. We expect valency[0] and valency[1]
 *  to be \pm 1.
 *
 *****************************************************************************/

static int test_charge1_set(psi_t * psi) {

  int ic, jc, kc, index;
  int nlocal[3];
  
  double rho0, rho1;

  coords_nlocal(nlocal);

  rho0 = 1.0 / (2.0*L(X)*L(Y));
  rho1 = 1.0 / (L(X)*L(Y)*(L(Z) - 2.0));

  /* Throughout set to rho1 */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi, index, 0.0);
	psi_rho_set(psi, index, 0, 0.0);
	psi_rho_set(psi, index, 1, rho1);
      }
    }
  }

  /* Now overwrite at the edges with rho0 */

  if (cart_coords(Z) == 0) {

    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = coords_index(ic, jc, kc);

	psi_rho_set(psi, index, 0, rho0);
	psi_rho_set(psi, index, 1, 0.0);
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = coords_index(ic, jc, kc);

	psi_rho_set(psi, index, 0, rho0);
	psi_rho_set(psi, index, 1, 0.0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_charge1_exact
 *
 *  Solve the tri-diagonal system appropriate for the 3-point stencil
 *  in one dimension (which is the z-direction). In parallel, all
 *  processes perform the whole solution.
 *
 *  The precise numerical solution is then obtained by solving the
 *  linear system.
 *
 *  We compare this with the solution obtained via the SOR function.
 *  Note that the linear system gives an answer which is different
 *  by a constant offset \psi_0. (I think this is related to the
 *  treatment of the periodic boundaries.)
 *
 *  The two solutions may then be compared to within (roughly) the
 *  relative tolerance prescribed for the SOR.
 *
 *  We also recompute the RHS by differencing the SOR solution to
 *  provide a final check.
 *
 *****************************************************************************/

static int test_charge1_exact(psi_t * obj, double tolerance) {

  int k, kp1, km1, index;
  int nlocal[3];
  int n;
  int ifail;

  double psi, psi0, rhotot, rhodiff;
  double * a = NULL;                   /* A is matrix for linear system */
  double * b = NULL;                   /* B is RHS / solution vector */
  double * c = NULL;                   /* Copy of the original RHS. */

  coords_nlocal(nlocal);

  assert(cart_size(Z) == 1);
  n = nlocal[Z];

  a = calloc(n*n, sizeof(double));
  b = calloc(n, sizeof(double));
  c = calloc(n, sizeof(double));
  if (a == NULL) fatal("calloc(a) failed\n");
  if (b == NULL) fatal("calloc(b) failed\n");
  if (c == NULL) fatal("calloc(c) failed\n");

  /* Set tridiagonal elements for periodic solution for the
   * three-point stencil. */

  for (k = 0; k < n; k++) {
    a[k*n + k] = -2.0;
    kp1 = k + 1;
    km1 = k - 1;
    if (k == 0) km1 = kp1;
    if (k == n-1) kp1 = km1;

    a[k*n + kp1] = 1.0;
    a[k*n + km1] = 1.0;
  }

  /* Set the right hand side and solve the linear system. */

  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, k + 1);
    psi_rho_elec(obj, index, b + k);
    c[k] = b[k];
  }

  ifail = util_gauss_jordan(n, a, b);
  assert(ifail == 0);

  rhotot = 0.0;
  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, 1+k);
    psi_psi(obj, index, &psi);
    if (k==0) psi0 = psi;
    rhodiff = obj->psi[index-1] - 2.0*obj->psi[index] + obj->psi[index+1];
    rhotot += c[k];

    assert(fabs(b[k] + psi0 - psi) < tolerance);
    assert(fabs(c[k] - rhodiff) < tolerance);
  }

  /* Total rho should be unchanged at zero. */
  assert(fabs(rhotot) < tolerance);

  free(c);
  free(b);
  free(a);

  return 0;
}

/*****************************************************************************
 *
 *  util_gauss_jordan
 *
 *  Solve linear system via Gauss Jordan elimination with full pivoting.
 *  See, e.g., Press et al page 39.
 *
 *  A is the n by n matrix, b is rhs on input and solution on output.
 *  We assume storage of A[i*n + j].
 *  A is column-scrambled inverse on exit. At the moment we don't bother
 *  to recover the inverse.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

int util_gauss_jordan(const int n, double * a, double * b) {

  int i, j, k, ia, ib;
  int irow, icol;
  int * ipivot = NULL;

  double rpivot, tmp;

  ipivot = calloc(n, sizeof(int));
  if (ipivot == NULL) return -3;

  icol = -1;
  irow = -1;

  for (j = 0; j < n; j++) {
    ipivot[j] = -1;
  }

  for (i = 0; i < n; i++) {
    tmp = 0.0;
    for (j = 0; j < n; j++) {
      if (ipivot[j] != 0) {
	for (k = 0; k < n; k++) {

	  if (ipivot[k] == -1) {
	    if (fabs(a[j*n + k]) >= tmp) {
	      tmp = fabs(a[j*n + k]);
	      irow = j;
	      icol = k;
	    }
	  }
	}
      }
    }

    assert(icol != -1);
    assert(irow != -1);

    ipivot[icol] += 1;

    if (irow != icol) {
      for (ia = 0; ia < n; ia++) {
	tmp = a[irow*n + ia];
	a[irow*n + ia] = a[icol*n + ia];
	a[icol*n + ia] = tmp;
      }
      tmp = b[irow];
      b[irow] = b[icol];
      b[icol] = tmp;
    }

    if (a[icol*n + icol] == 0.0) {
      free(ipivot);
      return -1;
    }

    rpivot = 1.0/a[icol*n + icol];
    a[icol*n + icol] = 1.0;

    for (ia = 0; ia < n; ia++) {
      a[icol*n + ia] *= rpivot;
    }
    b[icol] *= rpivot;

    for (ia = 0; ia < n; ia++) {
      if (ia != icol) {
	tmp = a[ia*n + icol];
	a[ia*n + icol] = 0.0;
	for (ib = 0; ib < n; ib++) {
	  a[ia*n + ib] -= a[icol*n + ib]*tmp;
	}
	b[ia] -= b[icol]*tmp;
      }
    }
  }

  /* Could recover the inverse here if required. */

  free(ipivot);

  return 0;
}
