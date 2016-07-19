/*****************************************************************************
 *
 *  test_phi_ch.c
 *
 *  Order parameter advection and the Cahn Hilliard equation.
 *
 *  In principle, this really is advection only; there is
 *  a default chemical potential (i.e., everywhere zero).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "leesedwards.h"
#include "hydro.h"
#include "map.h"
#include "field.h"
#include "gradient_3d_7pt_fluid.h"
#include "free_energy.h"
#include "phi_cahn_hilliard.h"
#include "util.h"
#include "tests.h"

static int test_u_zero(hydro_t * hydro, const double *);
static int test_advection(field_t * phi, hydro_t * hydro);
static int test_set_drop(field_t * phi, const double rc[3], double radius,
			 double xi0);
static int test_drop_difference(field_t * phi, const double rc[3],
				double radius, double xi0, double elnorm[3]);

/*****************************************************************************
 *
 *  test_ph_ch_suite
 *
 *****************************************************************************/

int test_phi_ch_suite(void) {

  int nf = 1;
  int nhalo = 2;

  hydro_t * hydro = NULL;
  field_t * phi = NULL;
  physics_t * phys = NULL;

  pe_init_quiet();
  coords_nhalo_set(nhalo);
  coords_init();
  physics_ref(&phys);
  le_init();

  field_create(nf, "phi", &phi);
  assert(phi);
  field_init(phi, nhalo);

  hydro_create(1, &hydro);
  assert(hydro);

  test_advection(phi, hydro);

  hydro_free(hydro);
  field_free(phi);

  info("PASS     ./unit/test_phi_ch\n");
  le_finish();
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_advection
 *
 *  Advect a droplet at constant velocity once across the lattice.
 *
 *****************************************************************************/

static int test_advection(field_t * phi, hydro_t * hydro) {

  int n, ntmax;
  double u[3];
  double rc[3];
  double r0;
  double xi0;
  double ell[3];

  assert(phi);
  assert(hydro);

  r0 = 0.25*L(X); /* Drop should fit */
  xi0 = 0.1*r0;   /* Cahn number = 1 / 10 (say) */

  /* Initial position is central */
  rc[X] = Lmin(X) + 0.5*L(X);
  rc[Y] = Lmin(Y) + 0.5*L(Y);
  rc[Z] = Lmin(Z) + 0.5*L(Z);

  u[X] = -0.25;
  u[Y] = 0.25;
  u[Z] = -0.25;

  ntmax = 10;

  test_u_zero(hydro, u);
  test_set_drop(phi, rc, r0, xi0);

  /* Steps */

  for (n = 0; n < ntmax; n++) {
    field_halo(phi);
    /* The map_t argument can be NULL here, as there is no solid;
     * the same is true for noise */
    phi_cahn_hilliard(NULL, phi, hydro, NULL, NULL);
  }

  /* Exact solution has position: */
  rc[X] += u[X]*ntmax;
  rc[Y] += u[Y]*ntmax;
  rc[Z] += u[Z]*ntmax;

  test_drop_difference(phi, rc, r0, xi0, ell);

  /* For these particular values... */
  assert(fabs(ell[0] - 1.1448194e-02) < FLT_EPSILON);
  assert(fabs(ell[1] - 1.5247605e-03) < FLT_EPSILON);
  assert(fabs(ell[2] - 2.5296108e-01) < FLT_EPSILON);

  return 0;
}

/****************************************************************************
 *
 *  test_u_zero
 *
 *  Set the velocity (and hence Courant numbers) uniformly in the
 *  system.
 *
 ****************************************************************************/

static int test_u_zero(hydro_t * hydro, const double u[3]) {

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
 *  Initialise a droplet with centre at rc, with given radius and
 *  interfacial width xi0.
 *
 *****************************************************************************/

static int test_set_drop(field_t * fphi, const double rc[3], double radius,
			 double xi0) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double rzeta; /* rxi ! */

  double phi, r;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  assert(fphi);
  rzeta = 1.0 / xi0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - rc[X];
	position[Y] = 1.0*(noffset[Y] + jc) - rc[Y];
	position[Z] = 1.0*(noffset[Z] + kc) - rc[Z];

	r = sqrt(dot_product(position, position));
	phi = tanh(rzeta*(r - radius));
	field_scalar_set(fphi, index, phi);
      }
    }
  }

  return 0;
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

static int test_drop_difference(field_t * fphi, const double rc[3],
				double radius, double xi0, double elnorm[3]) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double phi, phi0, r, rzeta, dphi;
  double ell_local[2], ell[2];
  double ell_inf_local, ell_inf;

  assert(fphi);
  assert(xi0 > 0.0);
  rzeta = 1.0 / xi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  ell_local[0]  = 0.0;
  ell_local[1]  = 0.0;
  ell_inf_local = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - rc[X];
	position[Y] = 1.0*(noffset[Y] + jc) - rc[Y];
	position[Z] = 1.0*(noffset[Z] + kc) - rc[Z];

	field_scalar(fphi, index, &phi);

	r = sqrt(dot_product(position, position));
	phi0 = tanh(rzeta*(r - radius));

	dphi = fabs(phi - phi0);
	ell_local[0] += fabs(phi - phi0);
	ell_local[1] += pow(phi - phi0, 2);
	if (dphi > ell_inf_local) ell_inf_local = dphi;

      }
    }
  }

  MPI_Allreduce(ell_local, ell, 2, MPI_DOUBLE, MPI_SUM, pe_comm());
  MPI_Allreduce(&ell_inf_local, &ell_inf, 1, MPI_DOUBLE, MPI_MAX, pe_comm());

  ell[0] /= L(X)*L(Y)*L(Z);
  ell[1] /= L(X)*L(Y)*L(Z);

  elnorm[0] = ell[0];
  elnorm[1] = ell[1];
  elnorm[2] = ell_inf;

  return 0;
}
