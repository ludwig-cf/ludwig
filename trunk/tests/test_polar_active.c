/*****************************************************************************
 *
 *  test_polar_active.c
 *
 *  Test of the polar active gel free energy against Davide's code.
 *  We check that the free energy density, the moleuclar field, and
 *  the stress are computed correctly for a given order parameter
 *  field.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinbrugh
 *
 *****************************************************************************/

#include <float.h>
#include <math.h>

#include "pe.h"
#include "tests.h"
#include "coords.h"
#include "leesedwards.h"
#include "phi.h"
#include "phi_gradients.h"
#include "gradient_2d_5pt_fluid.h"
#include "polar_active.h"

static void test_polar_active_aster(void);
static void test_polar_active_terms(void);
static void test_polar_active_init_aster(void);

/*****************************************************************************
 *
 *  main
 *
 *  This is a 2-d test in a system size 100 by 100.
 *
 *****************************************************************************/

int main (int argc, char ** argv) {

  int ntotal[3] = {100, 100, 1};

  MPI_Init(&argc, &argv);
  pe_init();

  coords_nhalo_set(2);
  coords_ntotal_set(ntotal);
  coords_init();
  le_init();

  phi_nop_set(3);
  phi_init();
  phi_gradients_dyadic_set(1);
  phi_gradients_init();
  gradient_2d_5pt_fluid_init();

  if (pe_size() == 1) {
    test_polar_active_aster();
    test_polar_active_terms();
  }
  else {
    info("Not running polar active tests in parallel yet.\n");
  }

  phi_gradients_finish();
  phi_finish();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_polar_active_aster
 *
 *  The order parameter represents an 'aster' configuration.
 *  All z-components should be zero in this 2-d configuration.
 *
 *****************************************************************************/

static void test_polar_active_aster(void) {

  int index;

  double fed;
  double p[3];
  double h[3];
  double s[3][3];

  polar_active_parameters_set(-0.1, +0.1, 0.01, 0.02);
  test_polar_active_init_aster();

  /* Order parameter */

  info("\nOrder parameter\n\n");

  index = coords_index(1, 1, 1);
  phi_vector(index, p);
  info("p_a(1, 1, 1) ...");
  test_assert(fabs(p[X] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  index = coords_index(2, 28, 1);
  phi_vector(index, p);
  info("p_a(2, 27, 1) ...");
  test_assert(fabs(p[X] - +9.0523694e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +4.2490714e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Gradient terms */

  phi_halo();
  phi_gradients_compute();

  /* Free energy density (not computed in independent code) */

  info("\nFree energy density\n\n");

  index = coords_index(1, 50, 1);
  fed = polar_active_free_energy_density(index);
  info("free energy density at (1, 50, 1) ...");
  test_assert(fabs(fed - -1.9979438e-02) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  index = coords_index(100, 3, 1);
  fed = polar_active_free_energy_density(index);
  info("free energy density at (100, 3, 1) ...");
  test_assert(fabs(fed - -1.2452523e-02) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  /* Molecular field */

  info("\nMolecular field\n\n");

  index = coords_index(4, 78, 1);
  polar_active_molecular_field(index, h);
  info("h_a(4, 78, 1) ...");
  test_assert(fabs(h[X] - -2.6571370e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - +1.5253651e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  index = coords_index(49, 49, 1);
  polar_active_molecular_field(index, h);
  info("h_a(49, 49, 1) ...");
  test_assert(fabs(h[X] - -8.8329259e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - -8.8329259e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Stress */

  info("\nStress\n\n");

  index = coords_index(3, 90, 1);
  polar_active_chemical_stress(index, s);
  info("s_ab(3, 90, 1) ...");
  test_assert(fabs(s[X][X] - +1.0398195e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +1.2809416e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +1.2784085e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +1.5748583e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  index = coords_index(100, 1, 1);
  polar_active_chemical_stress(index, s);
  info("s_ab(100, 1, 1) ...");
  test_assert(fabs(s[X][X] - +4.8979804e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +3.5860889e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -4.5958755e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +5.0000000e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  info("2-d test ok\n\n");

  return;
}

/*****************************************************************************
 *
 *  test_polar_active_terms
 *
 *  The order parameter and the molecular field are unchanged,
 *  but the stress changes with lambda, zeta non-zero.
 *
 *****************************************************************************/

void test_polar_active_terms(void) {

  int index;
  int ic, jc, kc;
  int nlocal[3];

  double s[3][3];

  coords_nlocal(nlocal);

  fe_v_lambda_set(2.1);
  polar_active_parameters_set(-0.1, +0.1, 0.01, 0.02);
  polar_active_zeta_set(0.001);

  test_polar_active_init_aster();

  phi_halo();
  phi_gradients_compute();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	polar_active_chemical_stress(index, s);

	test_assert(fabs(s[Y][Z] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[Z][Y] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[X][Z] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[Z][X] - 0.0) < TEST_DOUBLE_TOLERANCE);

      }
    }
  }

  index = coords_index(3, 90, 1);
  polar_active_chemical_stress(index, s);
  info("s_ab(3, 90, 1) ...");

  test_assert(fabs(s[X][X] - +2.5676086e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - -4.6394279e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -4.6394532e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +6.2712559e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - -3.1685874e-04) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  index = coords_index(100, 1, 1);
  polar_active_chemical_stress(index, s);

  info("s_ab(100, 1, 1) ...");
  test_assert(fabs(s[X][X] - -2.8683785e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +1.0485591e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +1.0403771e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - -3.1085746e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - +6.9667512e-02) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  info("Active stress ok\n\n");

  return;
}

/*****************************************************************************
 *
 *  test_polar_active_init_aster
 *
 *  A 2-d 'aster' configuration, where the vector points toward
 *  the centre of the 2-d system.
 *
 *****************************************************************************/

void test_polar_active_init_aster(void) {

  int nlocal[3];
  int ic, jc, kc, index;

  double p[3];
  double r;
  double x, y, z, x0, y0, z0;

  coords_nlocal(nlocal);
  /* coords_noffset(noffset);*/

  x0 = 0.5*L(X);
  y0 = 0.5*L(Y);
  z0 = 0.5*L(Z);

  if (nlocal[Z] == 1) z0 = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = 1.0*(ic-1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = 1.0*(jc-1);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = 1.0*(kc-1);

	p[X] = 0.0;
	p[Y] = 1.0;
	p[Z] = 0.0;

	r = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
	if (r > FLT_EPSILON) {
	  p[X] = -(x - x0)/r;
	  p[Y] = -(y - y0)/r;
	  p[Z] = -(z - z0)/r;
	}
	index = coords_index(ic, jc, kc);
	phi_vector_set(index, p);
      }
    }
  }

  return;
}
