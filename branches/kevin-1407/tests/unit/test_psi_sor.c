/*****************************************************************************
 *
 *  test_psi_sor.c
 *
 *  This is specifically SOR.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "control.h"
#include "psi_s.h"
#include "psi_sor.h"

#include "util.h"
#include "psi_stats.h"
#include "tests.h"

static int do_test_sor1(void);
static int test_charge1_set(psi_t * psi);
static int test_charge1_exact(psi_t * obj, f_vare_t fepsilon);

#define REF_PERMEATIVITY 1.0
static int fepsilon_constant(int index, double * epsilon);
static int fepsilon_sinz(int index, double * epsilon);

/*****************************************************************************
 *
 *  test_psi_sor_suite
 *
 *****************************************************************************/

int test_psi_sor_suite(void) {

  physics_t * phys = NULL;

  pe_init_quiet();
  physics_ref(&phys);

  control_time_set(-1); /* Kludge to avoid SOR iteration output */

  do_test_sor1();

  info("PASS     ./unit/test_psi_sor\n");
  pe_finalise();

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

   psi_t * psi = NULL;

  coords_nhalo_set(1);
  coords_init();

  psi_create(2, &psi);
  assert(psi);
  psi_valency_set(psi, 0, +1.0);
  psi_valency_set(psi, 1, -1.0);
  psi_epsilon_set(psi, REF_PERMEATIVITY);

  test_charge1_set(psi);
  psi_halo_psi(psi);
  psi_halo_rho(psi);
  psi_sor_poisson(psi);

  if (cart_size(Z) == 1) test_charge1_exact(psi, fepsilon_constant);

  /* Varying permeativity */

  test_charge1_set(psi);
  psi_halo_psi(psi);
  psi_halo_rho(psi);
  /* Following broken in latest vare solver */
  /* psi_sor_vare_poisson(psi, fepsilon_sinz);

  if (cart_size(Z) == 1) test_charge1_exact(psi, fepsilon_sinz);
  */
  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_charge1_set
 *
 *  Sets a uniform 'wall' charge at z = 1 and z = L_z and a uniform
 *  interior value elsewhere such that the system is overall charge
 *  neutral.
 *
 *  There is no sign, just a density. We expect valency[0] and valency[1]
 *  to be \pm 1.
 *
 *****************************************************************************/

static int test_charge1_set(psi_t * psi) {

  int nk;
  int ic, jc, kc, index;
  int nlocal[3];
  
  double rho0, rho1;

  double rho_min[4];  /* For psi_stats */
  double rho_max[4];  /* For psi_stats */
  double rho_tot[4];  /* For psi_stats */

  coords_nlocal(nlocal);

  rho0 = 1.0 / (2.0*L(X)*L(Y));           /* Edge values */
  rho1 = 1.0 / (L(X)*L(Y)*(L(Z) - 2.0));  /* Interior values */

  psi_nk(psi, &nk);
  assert(nk == 2);
  
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

  psi_stats_reduce(psi, rho_min, rho_max, rho_tot, 0, pe_comm());

  if (pe_rank() == 0) {
    /* psi all zero */
    assert(fabs(rho_min[0] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[0] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_tot[0] - 0.0) < DBL_EPSILON);
    /* First rho0 interior */
    assert(fabs(rho_min[1] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[1] - rho0) < DBL_EPSILON);
    assert(fabs(rho_tot[1] - 1.0) < DBL_EPSILON);
    /* Next rho1 edge */
    assert(fabs(rho_min[2] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[2] - rho1) < DBL_EPSILON);
    assert(fabs(rho_tot[2] - 1.0) < FLT_EPSILON);
    /* Total rho_elec */
    assert(fabs(rho_min[3] + rho1) < DBL_EPSILON); /* + because valency is - */
    assert(fabs(rho_max[3] - rho0) < DBL_EPSILON);
    assert(fabs(rho_tot[3] - 0.0) < FLT_EPSILON);
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
 *  by a constant offset \psi_0. (All solutions of the Poisson equation
 *  in periodic boundary conditions are the same to within an arbitrary
 *  constant provided the 'unit cell' is charge neutral.)
 *
 *  The two solutions may then be compared to within (roughly) the
 *  relative tolerance prescribed for the SOR. In turn, the solution
 *  of the Gauss Jordan routine has been checked agaisnt NAG F04AAF.
 *
 *  We also recompute the RHS by differencing the SOR solution with
 *  a three point stencil in one dimension to provide a final check.
 *
 *  For variable epsilon, described by the f_vare_t fepsilon,
 *  we set up a difference scheme using a three-point stencil:
 *
 *  e(i+1/2) psi(i+1) - [ e(i+1/2) + e(i-1/2) ] psi(i) + e(i-1/2) psi(i-1)
 *
 *  which is the same as that used in psi_cor.c and which collapses to the
 *  uniform case if e(r) is constant.
 *
 *****************************************************************************/

static int test_charge1_exact(psi_t * obj, f_vare_t fepsilon) {

  int k, kp1, km1, index;
  int nlocal[3];
  int n;
  int ifail;

  double * epsilon = NULL;             /* 1-d e = e(z) from fepsilon */
  double eph;                          /* epsilon(k + 1/2) */
  double emh;                          /* epsilon(k - 1/2) */
  double psi, psi0;                    /* Potential values */
  double tolerance;                    /* Absolute tolerance from psi_t */
  double rhotot;                       /* Charge conservation check */
  double rhodiff;                      /* Difference RHS check */

  double * a = NULL;                   /* A is matrix for linear system */
  double * b = NULL;                   /* B is RHS / solution vector */
  double * c = NULL;                   /* Copy of the original RHS. */

  coords_nlocal(nlocal);

  assert(cart_size(Z) == 1);
  n = nlocal[Z];

  /* Compute and store the permeativity values for convenience */

  epsilon = (double *) calloc(n, sizeof(double));
  if (epsilon == NULL) fatal("calloc(epsilon) failed\n");

  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, 1+k);
    fepsilon(index, epsilon + k);
  }

  /* Allocate space for exact solution */

  a = (double *) calloc(n*n, sizeof(double));
  b = (double *) calloc(n, sizeof(double));
  c = (double *) calloc(n, sizeof(double));
  if (a == NULL) fatal("calloc(a) failed\n");
  if (b == NULL) fatal("calloc(b) failed\n");
  if (c == NULL) fatal("calloc(c) failed\n");

  /* Set tridiagonal elements for periodic solution for the
   * three-point stencil. The logic is to remove the perioidic end
   * points which prevent a solution of the linear system. This
   * effectively sets a Dirichlet boundary condition with psi = 0
   * at both ends. */

  for (k = 0; k < n; k++) {
    
    kp1 = k + 1;
    km1 = k - 1;
    if (k == 0) km1 = kp1;
    if (k == n-1) kp1 = km1;

    eph = 0.5*(epsilon[k] + epsilon[kp1]);
    emh = 0.5*(epsilon[km1] + epsilon[k]);

    a[k*n + kp1] = eph;
    a[k*n + km1] = emh;
    a[k*n + k] = -(eph + emh);
  }

  /* Set the right hand side and solve the linear system. */

  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, k + 1);
    psi_rho_elec(obj, index, b + k);
    b[k] *= -1.0; /* Minus sign in RHS Poisson equation */

    c[k] = b[k];
  }

  ifail = util_gauss_jordan(n, a, b);
  assert(ifail == 0);

  /* Check the Gauss Jordan answer against the answer from psi_t */

  psi_abstol(obj, &tolerance);
  rhotot = 0.0;
  psi0 = 0.0;

  for (k = 0; k < n; k++) {
    index = coords_index(1, 1, 1+k);
    psi_psi(obj, index, &psi);
    if (k == 0) psi0 = psi;

    assert(fabs(b[k] + psi0 - psi) < tolerance);
    
    kp1 = k + 1;
    km1 = k - 1;
    if (k == 0) km1 = kp1;
    if (k == n-1) kp1 = km1;

    eph = 0.5*(epsilon[k] + epsilon[kp1]);
    emh = 0.5*(epsilon[km1] + epsilon[k]);
    rhodiff = emh*obj->psi[index-1] - (emh + eph)*obj->psi[index]
      + eph*obj->psi[index+1];

    assert(fabs(c[k] - rhodiff) < tolerance);
    rhotot += c[k];
  }

  /* Total rho should be unchanged at zero. */
  assert(fabs(rhotot) < tolerance);

  free(c);
  free(b);
  free(a);
  free(epsilon);

  return 0;
}

/*****************************************************************************
 *
 *  fepsilon_constant
 *
 *  Returns constant epsilon REF_PERMEATIVITY
 *
 *****************************************************************************/

static int fepsilon_constant(int index, double * epsilon) {

  assert(epsilon);

  *epsilon = REF_PERMEATIVITY;

  return 0;
}

/*****************************************************************************
 *
 *  fepsilon_sinz
 *
 *  Permeativity is a function of z only:
 *
 *    e = e0 sin(pi z / Lz)
 *
 *  The - 0.5 is to make it symmetric about the centre line.
 *
 *****************************************************************************/

static int fepsilon_sinz(int index, double * epsilon) {

  int coords[3];

  coords_index_to_ijk(index, coords);

  *epsilon = REF_PERMEATIVITY*sin(M_PI*(1.0*coords[Z] - 0.5)/L(Z));

  return 0;
}
