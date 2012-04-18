/*****************************************************************************
 *
 *  test_psi_sor.c
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include "pe.h"
#include "coords.h"
#include "psi_s.h"
#include "psi_sor.h"

static int do_test_sor1(void);
static int test_charge1_set(psi_t * psi);
static int test_charge1_exact(psi_t * obj);

int util_gauss_jordan(const int n, double * a, double * b);
static int util_gaussian(int n, double * a, double * xb);

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


  coords_nhalo_set(1);
  coords_init();
  psi_create(2, &psi_);
  psi_valency_set(psi_, 0, +1.0);
  psi_valency_set(psi_, 1, -1.0);

  test_charge1_set(psi_);
  psi_sor_poisson(psi_);
  if (cart_size(Z) == 1) test_charge1_exact(psi_);

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

	psi_rho_set(psi, index, 0, 0.0);
	psi_rho_set(psi, index, 1, rho1);
      }
    }
  }

  /* Now overwrite at the edges with rho0 */

  if (cart_coords(Z) == 0) {

    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 0; jc <= nlocal[Y]; jc++) {
	index = coords_index(ic, jc, kc);

	psi_rho_set(psi, index, 0, rho0);
	psi_rho_set(psi, index, 1, 0.0);
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 0; jc <= nlocal[Y]; jc++) {
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
 *****************************************************************************/

static int test_charge1_exact(psi_t * obj) {

  int k, kp1, km1, index;
  int nlocal[3];
  int n;
  int ifail;

  double psi, psi0, rhotot, rhodiff;
  double * a;
  double * b = NULL;
  double * c = NULL;

  coords_nlocal(nlocal);
  n = nlocal[Z];

  a = calloc(n*n, sizeof(double));
  b = calloc(n, sizeof(double));
  c = calloc(n, sizeof(double));
  if (a == NULL) fatal("calloc(a) failed\n");
  if (b == NULL) fatal("calloc(b) failed\n");
  if (c == NULL) fatal("calloc(c) failed\n");

  /* Set tridiagonal elements for periodic solution. */

  for (k = 0; k < n; k++) {
    a[k*n + k] = -2.0;
    kp1 = k + 1;
    km1 = k - 1;
    if (k == 0) km1 = kp1;
    if (k == n-1) kp1 = km1;

    a[k*n + kp1] = 1.0;
    a[k*n + km1] = 1.0;
  }

  /* Set the right hand side. */

  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, k + 1);
    psi_rho_elec(obj, index, b + k);
    /*b[k] = sin(2.0*4.0*atan(1.0)*(0.5 + 1.0*k)/64.0);*/
    c[k] = b[k];
  }

  ifail = util_gauss_jordan(n, a, b);

  rhotot = 0.0;
  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, 1+k);
    psi_psi(obj, index, &psi);
    if(k==0) psi0 = psi;
    rhodiff = obj->psi[index-1] - 2.0*obj->psi[index] + obj->psi[index+1];
    rhotot += c[k];
    info("%2d %14.7e %14.7e %14.7e %14.7e\n", k, b[k]+psi0, psi, c[k], rhodiff);
  }
  info("Gauss Jordan returned %d\n", ifail);
  info("Net charge: %14.7e\n", rhotot);

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
 *  A is column-scrambled inverse on exit.
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

    if (a[icol*n + icol] == 0.0) return -1;

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

/*****************************************************************************
 *
 *  util_gaussian
 *
 *  Solve linear system via Gaussian elimination. For the problems in this
 *  file, we only need to exchange rows, ie., have a partial pivot.
 *
 *  We solve Ax = b for A[NOP][NOP].
 *  xb is RHS on entry, and solution on exit.
 *  A is destroyed.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

static int util_gaussian(int n, double * a, double * xb) {

  int i, j, k;
  int ifail = 0;
  int iprow;
  int * ipivot;

  double tmp;

  ipivot = malloc(n*sizeof(int));
  if (ipivot == NULL) fatal("malloc(ipivot) failed\n");

  iprow = -1;
  for (k = 0; k < n; k++) {
    ipivot[k] = -1;
  }

  for (k = 0; k < n; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < n; i++) {
      if (ipivot[i] == -1) {
        if (fabs(a[i*n + k]) >= tmp) {
          tmp = fabs(a[i*n + k]);
          iprow = i;
        }
      }
    }
    ipivot[k] = iprow;

    /* divide pivot row by the pivot element a[iprow][k] */

    if (a[iprow*n + k] == 0.0) {
      fatal("Gaussian elimination failed in gradient calculation\n");
    }

    tmp = 1.0 / a[iprow*n + k];
    for (j = k; j < n; j++) {
      a[iprow*n + j] *= tmp;
    }
    xb[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < n; i++) {
      if (ipivot[i] == -1) {
        tmp = a[i*n + k];
        for (j = k; j < n; j++) {
          a[i*n + j] -= tmp*a[iprow*n + j];
        }
        xb[i] -= tmp*xb[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = n - 1; i > -1; i--) {
    iprow = ipivot[i];
    tmp = xb[iprow];
    for (k = i + 1; k < n; k++) {
      tmp -= a[iprow*n + k]*xb[ipivot[k]];
    }
    xb[iprow] = tmp;
  }

  free(ipivot);

  return ifail;
}
