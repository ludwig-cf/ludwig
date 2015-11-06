/*****************************************************************************
 *
 *  test_blue_phase.c
 *
 *  Tests for the blue phase free energy, molecular field, and
 *  the chemical stress.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field.h"
#include "field_grad.h"
#include "gradient_3d_7pt_fluid.h"
#include "gradient_3d_27pt_fluid.h"
#include "blue_phase.h"
#include "blue_phase_init.h"
#include "leesedwards.h"
#include "physics.h"
#include "tests.h"

static void multiply_gradient(double [3][3][3], double);
static void multiply_delsq(double [3][3], double);
static int test_o8m_struct(field_t * fq, field_grad_t * fqgrad);
static int test_bp_nonfield(void);

/*****************************************************************************
 *
 *  test_bp_suite
 *
 *****************************************************************************/

int test_bp_suite(void) {

  int nhalo = 2;
  field_t * fq = NULL;
  field_grad_t * fqgrad = NULL;

  pe_init_quiet();
  coords_nhalo_set(nhalo);
  coords_init();
  le_init(); /* Must be initialised to compute gradients. */

  test_bp_nonfield();

  field_create(NQAB, "q", &fq);
  field_init(fq, nhalo);
  field_grad_create(fq, 2, &fqgrad);
  field_grad_set(fqgrad, gradient_3d_27pt_fluid_d2, NULL);

  blue_phase_q_set(fq, fqgrad);

  test_o8m_struct(fq, fqgrad);

  field_grad_free(fqgrad);
  field_free(fq);
  le_finish();

  info("PASS     ./unit/test_blue_phase\n");
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_bp_nonfield
 *
 *  Values confimed via Wolfram Alpha eigenvalue widget.
 *
 *****************************************************************************/

static int test_bp_nonfield(void) {

  int ifail = 0;
  double q[3][3];
  double q5[NQAB] = {1.0, 1.0, 1.0, 1.0, 1.0};

  /* Artificial example */

  q[X][X] = 1.0;
  q[X][Y] = 2.0;
  q[X][Z] = 0.0;
  q[Y][X] = 2.0;
  q[Y][Y] = -1.0;
  q[Y][Z] = 1.5;
  q[Z][X] = 0.0;
  q[Z][Y] = 1.5;
  q[Z][Z] = 0.0;

  ifail = blue_phase_scalar_ops(q, q5);
  /*
  verbose("q5[0] = %14.7e\n", q5[0]);
  verbose("q5[1] = %14.7e\n", q5[1]);
  verbose("q5[2] = %14.7e\n", q5[2]);
  verbose("q5[3] = %14.7e\n", q5[3]);
  verbose("q5[4] = %14.7e\n", q5[4]);
  verbose("ifail = %d\n", ifail);
  */
  test_assert(fabs(q5[0] - 2.5214385 ) < FLT_EPSILON);
  test_assert(fabs(q5[1] - 0.74879672) < FLT_EPSILON);
  test_assert(fabs(q5[2] - 0.56962409) < FLT_EPSILON);
  test_assert(fabs(q5[3] - 0.33886852) < FLT_EPSILON);
  test_assert(fabs(q5[4] - 0.95411133) < FLT_EPSILON);

  return 0;
}

/*****************************************************************************
 *
 *  test_08m_struct
 *
 *  The test values here come from Davide's original code, of which
 *  we have a (more-or-less) independent implementation.
 *
 *  Note that the original code used 3d 7 point stencil for the
 *  gradient calculation. If the 27 point version is used in the
 *  tests, the resulting gradients must be adjusted to get the
 *  right answer. This is done by the multiply_gradient() and
 *  multiply_delsq() routines.
 *
 *  The parameters are:
 *
 *  Free energy:
 *       a0     = 0.014384711
 *       gamma  = 3.1764706
 *       q0     = numberhalftwists*nunitcell*sqrt(2)*pi/Ly
 *                number of half twists in unit cell = 1
 *                number of unit cells in pitch direction = 16 (Ly = 64)
 *       kappa0 = 0.01
 *       kappa1 = 0.01
 *                with one constant approximation (kappa0 = kappa1 = kappa).
 *       epsilon  dielectric anisotropy (typcially comes out to be 41.4 in
 *                lattice units based on matching Frederick transition).
 *                          
 *
 *  Molecular aspect ratio:
 *       xi = 0.7
 *
 *  The redshift should be 1.0.
 *
 *****************************************************************************/

int test_o8m_struct(field_t * fq, field_grad_t * fqgrad) {

  int nf;
  int ic, jc, kc, index;
  int numhalftwists = 1;
  int numunitcells = 16;
  int nlocal[3];

  double a0 = 0.014384711;
  double gamma = 3.1764706;
  double q0;
  double kappa = 0.01;
  double epsilon = 41.4;

  double xi = 0.7;
  double amplitude = -0.2;

  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double h[3][3];
  double field[3];
  double value;
  double e;
  physics_t * phys = NULL;

  physics_ref(&phys);
  coords_nlocal(nlocal);
  /*
  info("Blue phase O8M struct test\n");
  info("Must have q order parameter (nop = 5)...");
  */
  assert(fq);
  assert(fqgrad);
  field_nf(fq, &nf);
  test_assert(nf == NQAB);

  /* info("ok\n");*/

  q0 = sqrt(2.0)*4.0*atan(1.0)*numhalftwists*numunitcells / L(Y);
  blue_phase_set_free_energy_parameters(a0, gamma, kappa, q0);


  /* Check the chirality and the reduced temperature */

  value = sqrt(108.0*kappa/(a0*gamma))*q0;

  /* info("Testing chirality = %8.5f ...", value);*/
  test_assert(fabs(value - blue_phase_chirality()) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  value = 27.0*(1.0 - gamma/3.0)/gamma;

  /* info("Testing reduced temperature = %8.5f ...", value);*/
  test_assert(fabs(value - blue_phase_reduced_temperature())
	      < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  /* Set up the q tensor and sample some lattice sites. 
   * Note there are a limited number of unique order parameter values,
   * so an exhaustive test is probably not worth while. */

  blue_phase_init_amplitude_set(amplitude);
  blue_phase_O8M_init(fq);
  blue_phase_redshift_set(1.0);

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  /* info("Check q( 1, 1, 1)...");*/
  test_assert(fabs(q[X][X] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 2;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  /* info("Check q( 1, 1, 2)...");*/
  test_assert(fabs(q[X][X] - +0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.40000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.48284271247462) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 3;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  /* info("Check q( 1, 1, 3)...");*/
  test_assert(fabs(q[X][X] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - +0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] - +0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 12;
  kc = 4;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  /* info("Check q( 1,12, 4)...");*/
  test_assert(fabs(q[X][X] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - -0.08284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - +0.40000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  /* info("Check q( 2, 7, 6)...");*/
  test_assert(fabs(q[X][X] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.08284271247462) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  /* What we can test everywhere is that the q tensor is symmetric
   * and traceless. */

  /* info("Check q tensor is symmetric and traceless...");*/

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	field_tensor(fq, index, q);

	value = q[X][X] + q[Y][Y] + q[Z][Z];
	test_assert(fabs(value - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[X][Y] - q[Y][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[X][Z] - q[Z][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[Y][Z] - q[Z][Y]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* info("ok\n");*/


  /* Now the free energy density. This requires that the gradients are
   * set. These values use the standard 27-point stencil in 3-d. */

  /* info("Free energy density\n");*/

  blue_phase_set_xi(xi);
  field_halo(fq);
  field_grad_compute(fqgrad);

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  blue_phase_set_kernel_constants();
  void* vpcon=NULL;
  blue_phase_host_constant_ptr(&vpcon);
  bluePhaseKernelConstants_t * pcon = (bluePhaseKernelConstants_t *) vpcon;

  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 1, 1, 1)...");*/
  test_assert(fabs(value - 6.060508e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 2;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 1, 1, 2)...");*/
  test_assert(fabs(value - 1.056203e-02) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 3;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 1, 1, 3)...");*/
  test_assert(fabs(value - 6.060508e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 12;
  kc = 4;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 1,12, 4)...");*/
  test_assert(fabs(value - 6.609012e-04) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 2, 7, 6)...");*/
  test_assert(fabs(value - 6.609012e-04) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/



  /* Now the molecular field */

  /* info("Molecular field\n");*/

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 1, 1, 1)...");*/
  test_assert(fabs(h[X][X] - 0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] - 0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - 0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - 0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - 0.0171194) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/


  ic = 1;
  jc = 1;
  kc = 2;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 1, 1, 2)...");*/
  test_assert(fabs(h[X][X] - -0.0205178) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - +0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +0.0303829) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0323891) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 3;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 1, 1, 3)...");*/
  test_assert(fabs(h[X][X] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] - -0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - -0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0171194) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 12;
  kc = 4;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /*info("Check h( 1,12, 4)...");*/
  test_assert(fabs(h[X][X] - +0.0057295) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0023299) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - -0.0111454) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 2, 7, 6)...");*/
  test_assert(fabs(h[X][X] - +0.0054159) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +0.0057295) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0023299) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/


  /* info("Check molecular field tensor is symmetric...");*/

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	blue_phase_molecular_field(index, h);

	test_assert(fabs(h[X][Y] - h[Y][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(h[X][Z] - h[Z][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(h[Y][Z] - h[Z][Y]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* info("ok\n");*/


  /* Finally, the stress. This is not necessarily symmetric. */

  /* info("Thermodynamic contribution to stress\n");*/

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  blue_phase_compute_stress(q, dq, h, dsq, pcon);

  /* info("check s( 1, 1, 1)...");*/

  test_assert(fabs(dsq[X][X] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);  
  test_assert(fabs(dsq[Z][X] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/


  ic = 1;
  jc = 1;
  kc = 2;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  blue_phase_compute_stress(q, dq, h, dsq, pcon);

  /* info("check s( 1, 1, 2)...");*/
  test_assert(fabs(dsq[X][X] -  7.375082e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);  
  test_assert(fabs(dsq[Y][Y] - -4.179480e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -2.748179e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -2.871796e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -5.329164e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 1;
  kc = 3;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  blue_phase_compute_stress(q, dq, h, dsq, pcon);

  /* info("check s( 1, 1, 3)...");*/
  test_assert(fabs(dsq[X][X] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);  
  test_assert(fabs(dsq[Z][X] -  8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 1;
  jc = 12;
  kc = 4;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  blue_phase_compute_stress(q, dq, h, dsq, pcon);

  /* info("check s( 1,12, 4)...");*/
  test_assert(fabs(dsq[X][X] -  2.779621e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  7.180623e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  1.308445e-03) < TEST_FLOAT_TOLERANCE);  
  test_assert(fabs(dsq[Y][Y] - -5.056451e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -1.007305e-04) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  blue_phase_compute_stress(q, dq, h, dsq, pcon);

  /* info("check s( 2, 7, 6)...");*/
  test_assert(fabs(dsq[X][X] - -1.007305e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);  
  test_assert(fabs(dsq[Y][Y] -  2.779621e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] -  7.180623e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] -  1.308445e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -5.056451e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/


  /* Electric field test */

  /* Default electric field should be zero */

  value = blue_phase_dimensionless_field_strength();
  /* info("Dimensionless electric field is initially zero...");*/
  test_assert(fabs(value - 0.0) < TEST_DOUBLE_TOLERANCE);
  /* info("ok\n");*/

  field[X] = 0.0;
  field[Y] = 0.0;
  field[Z] = 1.0;

  physics_e0_set(field);
  blue_phase_dielectric_anisotropy_set(epsilon);

  e = sqrt(27.0*epsilon*1.0/(32.0*pi_*a0*gamma));

  value = blue_phase_dimensionless_field_strength();
  /* info("Electric field (0.0, 0.0, 1.0) gives dimensionless field %14.7e...", e);*/
  test_assert(fabs(value - e) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  field[X] = 1.0;
  field[Y] = 1.0;
  field[Z] = 1.0;

  physics_e0_set(field);

  e = sqrt(27.0*epsilon*3.0/(32.0*pi_*a0*gamma));

  value = blue_phase_dimensionless_field_strength();
  /* info("Electric field (1.0, 1.0, 1.0) gives dimensionless field %14.7e...", e);*/
  test_assert(fabs(value - e) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  /* Set dimensionless field to 0.2 for these parameters */

  field[X] = 0.012820969/sqrt(3.0);
  field[Y] = field[X];
  field[Z] = field[X];


  physics_e0_set(field);
  value = blue_phase_dimensionless_field_strength();
  /* info("Set dimensionless field 0.2...");*/
  test_assert(fabs(value - 0.2) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  /* Note the electric field remains switched on so... */
  blue_phase_set_kernel_constants();

  /* Check F(1,1,1) */

  field_halo(fq);
  field_grad_set(fqgrad, gradient_3d_7pt_fluid_d2, NULL);
  field_grad_compute(fqgrad);

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);

  field_grad_tensor_grad(fqgrad, index, dq);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 1, 1, 1)... %14.7e\n ", value);*/
  test_assert(fabs(value - 6.1626224e-03) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  value = blue_phase_compute_fed(q, dq, pcon);
  /* info("Check F( 2, 7, 6)... %14.7e ", value);*/
  test_assert(fabs(value - 6.7087074e-04) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/

  field[X] = 0.012820969;
  field[Y] = 0.0;
  field[Z] = 0.0;

  physics_e0_set(field);
  value = blue_phase_dimensionless_field_strength();
  /* info("Set dimensionless field again 0.2...");*/
  test_assert(fabs(value - 0.2) < TEST_FLOAT_TOLERANCE);
  /* info("ok\n");*/


  /* Note the electric field now changed so... */
  blue_phase_set_kernel_constants();

  ic = 1;
  jc = 1;
  kc = 1;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 1, 1, 1)...");*/

  test_assert(fabs(h[X][X] -  1.2034268e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - -6.0171338e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);

  /* info("ok\n");*/

  ic = 2;
  jc = 7;
  kc = 6;
  index = coords_index(ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);
  blue_phase_compute_h(q, dq, dsq, h, pcon);
  /* info("Check h( 2, 7, 6)...");*/

  test_assert(fabs(h[X][X] - +5.5362629e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000    ) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000    ) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +5.6693334e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +2.3299416e-03) < TEST_FLOAT_TOLERANCE);
  /*
  info("ok\n");

  info("Blue phase O8M structure ok\n");
  */
  return 0;
}

/*****************************************************************************
 *
 *  multiply_gradient
 *
 *  Adjust tensor gradient by fixed factor.
 *
 *****************************************************************************/

void multiply_gradient(double dq[3][3][3], double factor) {

  int ia, ib, ic;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	dq[ia][ib][ic] *= factor;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  multiply_delsq
 *
 *  Adjust tensor delsq by fixed factor.
 *
 *****************************************************************************/

void multiply_delsq(double dsq[3][3], double factor) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	dsq[ia][ib] *= factor;
    }
  }

  return;
}
