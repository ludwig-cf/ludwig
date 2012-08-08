/*****************************************************************************
 *
 *  test_phi_ch.c
 *
 *  Order parameter advection and the Cahn Hilliard equation.
 *
 *  In principle, this really is advection only; there is
 *  a default chemical potential (i.e., everywhere zero).
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "hydro.h"
#include "site_map.h"

#ifdef OLD_PHI
#include "phi.h"
#include "phi_gradients.h"
static void test_advection(hydro_t * hydro);
static void test_set_drop(void);
static void test_drop_difference(void);
#else
#include "field.h"
static int test_advection(field_t * phi, hydro_t * hydro);
static int test_set_drop(field_t * phi, double radius, double xi0);
static int test_drop_difference(field_t * phi, double radius, double xi0);
#endif

#include "gradient_3d_7pt_fluid.h"
#include "phi_cahn_hilliard.h"

#include "util.h"

static int test_u_zero(hydro_t * hydro, const double *);
#ifdef OLD_PHI
int main (int argc, char ** argv) {

  int nf = 1;
  int nhalo = 2;
  hydro_t * hydro = NULL;

  MPI_Init(&argc, &argv);
  pe_init();
  coords_nhalo_set(nhalo);
  coords_init();
  le_init();
  site_map_init();

  phi_nop_set(nf);
  phi_init();
  phi_gradients_init();
  gradient_3d_7pt_fluid_init();

  hydro_create(1, &hydro);
  test_advection(hydro);

  hydro_free(hydro);
  phi_finish();

  le_finish();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}
#else
int main (int argc, char ** argv) {

  int nf = 1;
  int nhalo = 2;
  hydro_t * hydro = NULL;
  field_t * phi = NULL;

  MPI_Init(&argc, &argv);
  pe_init();
  coords_nhalo_set(nhalo);
  coords_init();
  le_init();
  site_map_init();

  field_create(nf, "phi", &phi);
  field_init(phi, nhalo);

  hydro_create(1, &hydro);
  test_advection(phi, hydro);

  hydro_free(hydro);
  field_free(phi);

  le_finish();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}
#endif

/*****************************************************************************
 *
 *  test_advection
 *
 *  Advect a droplet at constant velocity once across the lattice.
 *
 *****************************************************************************/
#ifdef OLD_PHI
void test_advection(hydro_t * hydro) {

  int n, ntmax;
  double u[3];

  assert(hydro);

  u[X] = -0.25;
  u[Y] = 0.25;
  u[Z] = -0.25;
  ntmax = -L(X)/u[X];
  info("Time steps: %d\n", ntmax);

  test_u_zero(hydro, u);
  test_set_drop();

  /* Steps */

  for (n = 0; n < ntmax; n++) {
    phi_halo();
    phi_gradients_compute();
    phi_cahn_hilliard(hydro);
  }

  test_drop_difference();

  return;
}
#else
int test_advection(field_t * phi, hydro_t * hydro) {

  int n, ntmax;
  double u[3];
  double r0;
  double xi0;

  assert(phi);
  assert(hydro);

  r0 = 0.25*L(X); /* Drop should fit */
  xi0 = 0.1*r0;   /* Cahn number = 1 / 10 (say) */

  u[X] = -0.25;
  u[Y] = 0.25;
  u[Z] = -0.25;
  ntmax = -L(X)/u[X];
  info("Time steps: %d\n", ntmax);

  test_u_zero(hydro, u);
  test_set_drop(phi, r0, xi0);

  /* Steps */

  for (n = 0; n < ntmax; n++) {
    field_halo(phi);
    phi_cahn_hilliard(phi, hydro);
  }

  test_drop_difference(phi, r0, xi0);

  return 0;
}
#endif

/****************************************************************************
 *
 *  test_u_zero
 *
 *  Set the velocity (and hence Courant numbers) uniformly in the
 *  system.
 *
 ****************************************************************************/

int test_u_zero(hydro_t * hydro, const double u[3]) {

  int ic, jc, kc, index;
  int nlocal[3];

  assert(hydro);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	hydro_u_set(hydro, index, u);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_set_drop
 *
 *  Initialise a droplet at radius = L/4 in the centre of the system.
 *
 *****************************************************************************/

#ifdef OLD_PHI
void test_set_drop() {
#else
  int test_set_drop(field_t * fphi, double radius, double xi0) {
#endif

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
#ifdef OLD_PHI
  double rzeta;
#else
  double rzeta; /* rxi ! */
#endif

  double phi, r;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
#ifdef OLD_PHI
  radius = 0.25*L(X);
  rzeta = 1.0/symmetric_interfacial_width();
#else
  assert(fphi);
  rzeta = 1.0 / xi0;
#endif

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - (0.5*L(X) + Lmin(X));
	position[Y] = 1.0*(noffset[Y] + jc) - (0.5*L(Y) + Lmin(Y));
	position[Z] = 1.0*(noffset[Z] + kc) - (0.5*L(Z) + Lmin(Z));

	r = sqrt(dot_product(position, position));
	phi = tanh(rzeta*(r - radius));
#ifdef OLD_PHI
	phi_set_phi_site(index, phi);
#else
	field_scalar_set(fphi, index, phi);
#endif

      }
    }
  }
#ifdef OLD_PHI
  return;
#else
  return 0;
#endif
}

/*****************************************************************************
 *
 *  test_drop_difference
 *
 *  For the advection only problem, the exact solution is known (the
 *  centre of the drop is just displaced u\Delta t). So we can work
 *  out the errors.
 *
 *****************************************************************************/
#ifdef OLD_PHI
void test_drop_difference() {
#else
  int test_drop_difference(field_t * fphi, double radius, double xi0) {
#endif
  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double phi, phi0, r, rzeta, dphi, dmax;
  double sum[2]; 
#ifdef OLD_PHI
  double radius;

  radius = 0.25*L(X);
  rzeta = 1.0/symmetric_interfacial_width();
#else
  assert(fphi);
  assert(xi0 > 0.0);
  rzeta = 1.0 / xi0;
#endif
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  sum[0] = 0.0;
  sum[1] = 0.0;
  dmax   = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - (0.5*L(X)+Lmin(X));
	position[Y] = 1.0*(noffset[Y] + jc) - (0.5*L(Y)+Lmin(Y));
	position[Z] = 1.0*(noffset[Z] + kc) - (0.5*L(Z)+Lmin(Z));
#ifdef OLD_PHI
	phi = phi_get_phi_site(index);
#else
	field_scalar(fphi, index, &phi);
#endif
	r = sqrt(dot_product(position, position));
	phi0 = tanh(rzeta*(r - radius));

	dphi = fabs(phi - phi0);
	sum[0] += fabs(phi - phi0);
	sum[1] += pow(phi - phi0, 2);
	if (dphi > dmax) dmax = dphi;

      }
    }
  }

  sum[0] /= L(X)*L(Y)*L(Z);
  sum[1] /= L(X)*L(Y)*L(Z);

  info("Norms L1 = %14.7e L2 = %14.7e L\\inf = %14.7e\n",
       sum[0], sum[1], dmax);
#ifdef OLD_PHI
  return;
#else
  assert(fabs(sum[0] - 1.0434109e-01) < FLT_EPSILON);
  assert(fabs(sum[1] - 5.4361537e-02) < FLT_EPSILON);
  assert(fabs(dmax   - 1.1392812e+00) < FLT_EPSILON);

  return 0;
#endif
}
