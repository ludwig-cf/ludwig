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

#include "field.h"
#include "field_grad.h"
#include "gradient_2d_5pt_fluid.h"
#include "polar_active.h"

static int test_polar_active_aster(field_t * fp, field_grad_t * fpgrad);
static int test_polar_active_terms(field_t * fp, field_grad_t * fpgrad);
static int test_polar_active_init_aster(field_t * fp);

/*****************************************************************************
 *
 *  main
 *
 *  This is a 2-d test in a system size 100 by 100.
 *
 *****************************************************************************/

int main (int argc, char ** argv) {

  int nf = NVECTOR;
  int nhalo = 2;
  int ntotal[3] = {100, 100, 1};

  field_t * fp = NULL;
  field_grad_t * fpgrad = NULL;

  MPI_Init(&argc, &argv);
  pe_init();

  coords_nhalo_set(nhalo);
  coords_ntotal_set(ntotal);
  coords_init();
  le_init();

  field_create(nf, "p", &fp);
  field_init(fp, nhalo);
  field_grad_create(fp, 2, &fpgrad);
  field_grad_set(fpgrad, gradient_2d_5pt_fluid_d2, NULL);
  polar_active_p_set(fp, fpgrad);

  if (pe_size() == 1) {
    test_polar_active_aster(fp, fpgrad);
    test_polar_active_terms(fp, fpgrad);
  }
  else {
    info("Not running polar active tests in parallel yet.\n");
  }

  field_grad_free(fpgrad);
  field_free(fp);

  le_finish();
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

static int test_polar_active_aster(field_t * fp, field_grad_t * fpgrad) {

  int index;

  double fed;
  double p[3];
  double h[3];
  double s[3][3];

  /* Note that the k2 = 0.02 here is not effective, as all the terms
   * the the polar active are not currently compluted. If all terms
   * were present, the relevant results would be changed. */

  polar_active_parameters_set(-0.1, +0.1, 0.01, 0.02);
  test_polar_active_init_aster(fp);

  /* Order parameter */

  info("\nOrder parameter\n\n");

  index = coords_index(1, 1, 1);
  field_vector(fp, index, p);

  info("p_a(1, 1, 1) ...");
  test_assert(fabs(p[X] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  index = coords_index(2, 28, 1);
  field_vector(fp, index, p);

  info("p_a(2, 27, 1) ...");
  test_assert(fabs(p[X] - +9.0523694e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +4.2490714e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Gradient terms */

  field_halo(fp);
  field_grad_compute(fpgrad);

  /* Free energy density (not computed in independent code) */

  info("\nFree energy density\n\n");

  index = coords_index(1, 50, 1);
  fed = polar_active_free_energy_density(index);
  info("free energy density at (1, 50, 1) ...");
  info("ok\n");

  index = coords_index(100, 3, 1);
  fed = polar_active_free_energy_density(index);
  info("free energy density at (100, 3, 1) ...");
  test_assert(fabs(fed - -2.2448448e-02) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  /* Molecular field */

  info("\nMolecular field\n\n");

  index = coords_index(4, 78, 1);
  polar_active_molecular_field(index, h);
  info("h_a(4, 78, 1) ...");
  test_assert(fabs(h[X] - -2.9526261e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - +1.6947361e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  index = coords_index(49, 49, 1);
  polar_active_molecular_field(index, h);
  info("h_a(49, 49, 1) ...");
  test_assert(fabs(h[X] - -1.0003585e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - -1.0003585e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Stress */

  info("\nStress\n\n");

  index = coords_index(3, 90, 1);
  polar_active_chemical_stress(index, s);
  info("s_ab(3, 90, 1) ...");
  test_assert(fabs(s[X][X] - +1.0398195e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +1.2798462e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +1.2795039e-06) < TEST_FLOAT_TOLERANCE);
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
  test_assert(fabs(s[X][Y] - -4.9469398e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -5.1509267e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +5.0000000e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  info("2-d test ok\n\n");

  return 0;
}

/*****************************************************************************
 *
 *  test_polar_active_terms
 *
 *  The order parameter and the molecular field are unchanged,
 *  but the stress changes with lambda, zeta non-zero.
 *
 *****************************************************************************/

int test_polar_active_terms(field_t * fp, field_grad_t * fpgrad) {

  int index;
  int ic, jc, kc;
  int nlocal[3];

  double s[3][3];

  coords_nlocal(nlocal);

  fe_v_lambda_set(2.1);
  polar_active_parameters_set(-0.1, +0.1, 0.01, 0.02);
  polar_active_zeta_set(0.001);

  test_polar_active_init_aster(fp);
  field_halo(fp);
  field_grad_compute(fpgrad);

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

  test_assert(fabs(s[X][X] - +2.6858170e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - -4.8544429e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -4.8544463e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +6.5535744e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - -3.3150277e-04) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  index = coords_index(100, 1, 1);
  polar_active_chemical_stress(index, s);

  info("s_ab(100, 1, 1) ...");
  test_assert(fabs(s[X][X] - -1.5237375e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +2.0447484e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +2.0445444e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - -2.2456775e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - +1.3667395e-02) < TEST_FLOAT_TOLERANCE);
  info("ok\n");

  info("Active stress ok\n\n");

  return 0;
}

/*****************************************************************************
 *
 *  test_polar_active_init_aster
 *
 *  A 2-d 'aster' configuration, where the vector points toward
 *  the centre of the 2-d system.
 *
 *****************************************************************************/

int test_polar_active_init_aster(field_t * fp) {

  int nlocal[3];
  int ic, jc, kc, index;

  double p[3];
  double r;
  double x, y, z, x0, y0, z0;

  coords_nlocal(nlocal);

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
	field_vector_set(fp, index, p);
      }
    }
  }

  return 0;
}
