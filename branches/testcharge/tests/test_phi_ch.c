/*****************************************************************************
 *
 *  test_phi_ch.c
 *
 *  Order parameter advection and the Cahn Hilliard equation.
 *  NOT A COMPLETE TEST.
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
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "hydro.h"
#include "site_map.h"

#include "phi.h"
#include "phi_gradients.h"
#include "gradient_3d_7pt_fluid.h"
#include "phi_cahn_hilliard.h"
#include "symmetric.h"

#include "util.h"

static void test_advection(hydro_t * hydro);
static int test_u_zero(hydro_t * hydro, const double *);
static void test_set_drop(void);
static void test_drop_difference(void);

int main (int argc, char ** argv) {

  hydro_t * hydro;

  MPI_Init(&argc, &argv);
  pe_init();
  coords_nhalo_set(2);
  coords_init();
  le_init();
  site_map_init();
  phi_nop_set(1);
  phi_init();
  phi_gradients_init();
  gradient_3d_7pt_fluid_init();
  hydro_create(1, &hydro);

  test_advection(hydro);

  hydro_free(hydro);
  phi_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_advection
 *
 *  Advect a droplet at constant velocity once across the lattice.
 *
 *****************************************************************************/

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

void test_set_drop() {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double phi, r, radius, rzeta;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  radius = 0.25*L(X);
  rzeta = 1.0/symmetric_interfacial_width();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - (0.5*L(X) + Lmin(X));
	position[Y] = 1.0*(noffset[Y] + jc) - (0.5*L(Y) + Lmin(Y));
	position[Z] = 1.0*(noffset[Z] + kc) - (0.5*L(Z) + Lmin(Z));

	r = sqrt(dot_product(position, position));
	phi = tanh(rzeta*(r - radius));
	phi_set_phi_site(index, phi);

      }
    }
  }

  return;
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

void test_drop_difference() {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double phi, phi0, r, radius, rzeta, dphi, dmax;
  double sum[2]; 

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  radius = 0.25*L(X);
  rzeta = 1.0/symmetric_interfacial_width();

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

	phi = phi_get_phi_site(index);

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

  info("Norms L1 = %g L2 = %g L\\inf = %g\n", sum[0], sum[1], dmax);

  return;
}
