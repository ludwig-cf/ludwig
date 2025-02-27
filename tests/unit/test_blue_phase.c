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
 *  (c) 2010-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "kernel.h"
#include "leesedwards.h"
#include "physics.h"
#include "tests.h"

static void multiply_gradient(double [3][3][3], double);
static void multiply_delsq(double [3][3], double);
static int test_o8m_struct(pe_t * pe, cs_t * cs, lees_edw_t * le, fe_lc_t * fe,
			   field_t * fq,
			   field_grad_t * fqgrad);
static int test_bp_nonfield(void);
int test_fe_lc_dimensionless_field_strength(pe_t * pe);


__host__ int do_test_fe_lc_device1(pe_t * pe, cs_t * cs, fe_lc_t * fe);
__global__ void do_test_fe_lc_kernel1(fe_lc_t * fe, const fe_lc_param_t * ref);


/*****************************************************************************
 *
 *  test_bp_suite
 *
 *****************************************************************************/

int test_bp_suite(void) {

  int nhalo = 2;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;
  field_t * fq = NULL;
  field_grad_t * fqgrad = NULL;
  fe_lc_t * fe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);
  lees_edw_create(pe, cs, NULL, &le);

  test_bp_nonfield();
  test_fe_lc_dimensionless_field_strength(pe);

  {
    field_options_t opts = field_options_ndata_nhalo(NQAB, nhalo);
    field_create(pe, cs, le, "q", &opts, &fq);

    field_grad_create(pe, fq, 2, &fqgrad);
    field_grad_set(fqgrad, grad_3d_27pt_fluid_d2, NULL);
  }

  fe_lc_create(pe, cs, le, fq, fqgrad, &fe);

  test_o8m_struct(pe, cs, le, fe, fq, fqgrad);
  do_test_fe_lc_device1(pe, cs, fe);

  fe_lc_free(fe);
  field_grad_free(fqgrad);
  field_free(fq);
  lees_edw_free(le);
  cs_free(cs);

  pe_info(pe, "PASS     ./unit/test_blue_phase\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_bp_nonfield
 *
 *  Values confirmed via Wolfram Alpha eigenvalue widget.
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

  ifail = fe_lc_scalar_ops(q, q5);

  test_assert(ifail == 0);
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
 *       epsilon  dielectric anisotropy (typically comes out to be 41.4 in
 *                lattice units based on matching Frederick transition).
 *
 *
 *  Molecular aspect ratio:
 *       xi = 0.7
 *
 *  The redshift should be 1.0.
 *
 *****************************************************************************/

int test_o8m_struct(pe_t * pe, cs_t * cs, lees_edw_t * le, fe_lc_t * fe,
		    field_t * fq,
		    field_grad_t * fqgrad) {

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
  double value, vtest;
  double ltot[3];
  fe_lc_param_t param = {0};
  physics_t * phys = NULL;

  assert(pe);
  assert(le);

  physics_create(pe, &phys);

  lees_edw_ltot(le, ltot);
  lees_edw_nlocal(le, nlocal);

  /* "Blue phase O8M struct test"
   * "Must have q order parameter (nop = 5)..." */

  assert(fq);
  assert(fqgrad);
  field_nf(fq, &nf);
  test_assert(nf == NQAB);

  q0 = sqrt(2.0)*4.0*atan(1.0)*numhalftwists*numunitcells / ltot[Y];

  param.a0 = a0;
  param.gamma = gamma;
  param.kappa0 = kappa;
  param.kappa1 = kappa;
  param.q0 = q0;
  param.xi = xi;
  param.amplitude0 = amplitude;
  param.redshift = 1.0;
  param.rredshift = 1.0;
  param.epsilon = epsilon;
  fe_lc_param_set(fe, &param);
  fe_lc_param_commit(fe);

  /* Check the chirality and the reduced temperature */

  value = sqrt(108.0*kappa/(a0*gamma))*q0;

  fe_lc_chirality(fe, &vtest);
  test_assert(fabs(value - vtest) < TEST_DOUBLE_TOLERANCE);

  value = 27.0*(1.0 - gamma/3.0)/gamma;

  fe_lc_reduced_temperature(fe, &vtest);
  test_assert(fabs(value - vtest) < TEST_DOUBLE_TOLERANCE);

  /* Set up the q tensor and sample some lattice sites.
   * Note there are a limited number of unique order parameter values,
   * so an exhaustive test is probably not worth while. */

  {
    double angles[3] = {0}; /* No rotation. */
    blue_phase_O8M_init(cs, &param, fq, angles);
  }

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  test_assert(fabs(q[X][X] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 2;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  test_assert(fabs(q[X][X] - +0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.40000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.48284271247462) < TEST_DOUBLE_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 3;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  test_assert(fabs(q[X][X] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - +0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] - +0.28284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.28284271247462) < TEST_DOUBLE_TOLERANCE);

  ic = 1;
  jc = 12;
  kc = 4;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  test_assert(fabs(q[X][X] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] - -0.08284271247462) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - +0.40000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  test_assert(fabs(q[X][X] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Y] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[X][Z] -  0.00000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Y] - -0.20000000000000) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(q[Y][Z] - -0.08284271247462) < TEST_DOUBLE_TOLERANCE);

  /* What we can test everywhere is that the q tensor is symmetric
   * and traceless. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = lees_edw_index(le, ic, jc, kc);
	field_tensor(fq, index, q);

	value = q[X][X] + q[Y][Y] + q[Z][Z];
	test_assert(fabs(value - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[X][Y] - q[Y][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[X][Z] - q[Z][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(q[Y][Z] - q[Z][Y]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }


  /* Now the free energy density. This requires that the gradients are
   * set. These values use the standard 27-point stencil in 3-d. */

  /* Gradient computation is on the device, so ...
  *  ... copy to gpu and perform gpu halo ... */

  field_memcpy(fq, tdpMemcpyHostToDevice);
  field_halo(fq);

  field_grad_compute(fqgrad);
  field_grad_memcpy(fqgrad, tdpMemcpyDeviceToHost);

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 6.060508e-03) < TEST_FLOAT_TOLERANCE);

  {
    double fed_bulk = 0.0;
    double fed_grad = 0.0;

    fe_lc_compute_bulk_fed(fe, q, &fed_bulk);
    fe_lc_compute_gradient_fed(fe, q, dq, &fed_grad);
    assert(fabs(value - (fed_bulk + fed_grad)) < DBL_EPSILON);
  }

  ic = 1;
  jc = 1;
  kc = 2;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 1.056203e-02) < TEST_FLOAT_TOLERANCE);

  {
    double fed_bulk = 0.0;
    double fed_grad = 0.0;

    fe_lc_compute_bulk_fed(fe, q, &fed_bulk);
    fe_lc_compute_gradient_fed(fe, q, dq, &fed_grad);
    assert(fabs(value - (fed_bulk + fed_grad)) < DBL_EPSILON);
  }

  ic = 1;
  jc = 1;
  kc = 3;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 6.060508e-03) < TEST_FLOAT_TOLERANCE);

  {
    double fed_bulk = 0.0;
    double fed_grad = 0.0;

    fe_lc_compute_bulk_fed(fe, q, &fed_bulk);
    fe_lc_compute_gradient_fed(fe, q, dq, &fed_grad);
    assert(fabs(value - (fed_bulk + fed_grad)) < DBL_EPSILON);
  }

  ic = 1;
  jc = 12;
  kc = 4;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 6.609012e-04) < TEST_FLOAT_TOLERANCE);

  {
    double fed_bulk = 0.0;
    double fed_grad = 0.0;

    fe_lc_compute_bulk_fed(fe, q, &fed_bulk);
    fe_lc_compute_gradient_fed(fe, q, dq, &fed_grad);
    assert(fabs(value - (fed_bulk + fed_grad)) < DBL_EPSILON);
  }

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  multiply_gradient(dq, 3.0);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 6.609012e-04) < TEST_FLOAT_TOLERANCE);

  {
    double fed_bulk = 0.0;
    double fed_grad = 0.0;

    fe_lc_compute_bulk_fed(fe, q, &fed_bulk);
    fe_lc_compute_gradient_fed(fe, q, dq, &fed_grad);
    assert(fabs(value - (fed_bulk + fed_grad)) < DBL_EPSILON);
  }



  /* Now the molecular field */

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] - 0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] - 0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - 0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - 0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - 0.0171194) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 2;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] - -0.0205178) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - +0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +0.0303829) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0323891) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 3;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] - -0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] - -0.0171194) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0171194) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 12;
  kc = 4;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] - +0.0057295) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0023299) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - -0.0111454) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] - +0.0054159) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +0.0057295) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +0.0023299) < TEST_FLOAT_TOLERANCE);


  /* Check molecular field tensor is symmetric... */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = lees_edw_index(le, ic, jc, kc);
	fe_lc_mol_field(fe, index, h);

	test_assert(fabs(h[X][Y] - h[Y][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(h[X][Z] - h[Z][X]) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(h[Y][Z] - h[Z][Y]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  /* Finally, the stress. This is not necessarily symmetric. */

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, dsq);

  test_assert(fabs(dsq[X][X] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 2;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, dsq);

  test_assert(fabs(dsq[X][X] -  7.375082e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -4.179480e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -2.748179e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -2.871796e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -5.329164e-03) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 1;
  kc = 3;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, dsq);

  test_assert(fabs(dsq[X][X] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] - -8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  8.924220e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] - -9.837494e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -7.887056e-03) < TEST_FLOAT_TOLERANCE);

  ic = 1;
  jc = 12;
  kc = 4;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, dsq);

  test_assert(fabs(dsq[X][X] -  2.779621e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  7.180623e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  1.308445e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] - -5.056451e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -1.007305e-04) < TEST_FLOAT_TOLERANCE);

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  multiply_gradient(dq, 3.0);
  multiply_delsq(dsq, 1.5);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);
  fe_lc_compute_stress(fe, q, dq, h, dsq);

  test_assert(fabs(dsq[X][X] - -1.007305e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Y] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[X][Z] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Y] -  2.779621e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Y][Z] -  7.180623e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][X] -  0.0000000000) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Y] -  1.308445e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(dsq[Z][Z] - -5.056451e-03) < TEST_FLOAT_TOLERANCE);

  /* Now do that again without doctoring the values of the gradients
   * to test the decomposition into bulk and gradient parts */

  {
    double sfull[3][3] = {0};
    double sbulk[3][3] = {0};
    double sgrad[3][3] = {0};
    ic = 1;
    jc = 1;
    kc = 1;
    index = lees_edw_index(le, ic, jc, kc);

    fe_lc_stress(fe, index, sfull);
    fe_lc_bulk_stress(fe, index, sbulk);
    fe_lc_grad_stress(fe, index, sgrad);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(sfull[ia][ib] - (sbulk[ia][ib] + sgrad[ia][ib]))
	       < DBL_EPSILON);
      }
    }
  }

  {
    double sfull[3][3] = {0};
    double sbulk[3][3] = {0};
    double sgrad[3][3] = {0};
    ic = 1;
    jc = 1;
    kc = 2;
    index = lees_edw_index(le, ic, jc, kc);

    fe_lc_stress(fe, index, sfull);
    fe_lc_bulk_stress(fe, index, sbulk);
    fe_lc_grad_stress(fe, index, sgrad);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(sfull[ia][ib] - (sbulk[ia][ib] + sgrad[ia][ib]))
	       < DBL_EPSILON);
      }
    }
  }

  {
    double sfull[3][3] = {0};
    double sbulk[3][3] = {0};
    double sgrad[3][3] = {0};
    ic = 1;
    jc = 1;
    kc = 3;
    index = lees_edw_index(le, ic, jc, kc);

    fe_lc_stress(fe, index, sfull);
    fe_lc_bulk_stress(fe, index, sbulk);
    fe_lc_grad_stress(fe, index, sgrad);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(sfull[ia][ib] - (sbulk[ia][ib] + sgrad[ia][ib]))
	       < DBL_EPSILON);
      }
    }
  }

  {
    double sfull[3][3] = {0};
    double sbulk[3][3] = {0};
    double sgrad[3][3] = {0};
    ic = 1;
    jc = 12;
    kc = 4;
    index = lees_edw_index(le, ic, jc, kc);

    fe_lc_stress(fe, index, sfull);
    fe_lc_bulk_stress(fe, index, sbulk);
    fe_lc_grad_stress(fe, index, sgrad);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(sfull[ia][ib] - (sbulk[ia][ib] + sgrad[ia][ib]))
	       < DBL_EPSILON);
      }
    }
  }

  {
    double sfull[3][3] = {0};
    double sbulk[3][3] = {0};
    double sgrad[3][3] = {0};
    ic = 2;
    jc = 6;
    kc = 7;
    index = lees_edw_index(le, ic, jc, kc);

    fe_lc_stress(fe, index, sfull);
    fe_lc_bulk_stress(fe, index, sbulk);
    fe_lc_grad_stress(fe, index, sgrad);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(sfull[ia][ib] - (sbulk[ia][ib] + sgrad[ia][ib]))
	       < DBL_EPSILON);
      }
    }
  }

  /* Electric field test */

  /* Set dimensionless field to 0.2 for these parameters */

  field[X] = 0.012820969/sqrt(3.0);
  field[Y] = field[X];
  field[Z] = field[X];

  fe->param->e0[X] = field[X];
  fe->param->e0[Y] = field[Y];
  fe->param->e0[Z] = field[Z];

  /* Note the electric field remains switched on so... */

  field_grad_set(fqgrad, grad_3d_7pt_fluid_d2, NULL);

  field_memcpy(fq, tdpMemcpyHostToDevice);
  field_halo(fq);

  field_grad_compute(fqgrad);

  field_grad_memcpy(fqgrad, tdpMemcpyDeviceToHost);

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);

  field_grad_tensor_grad(fqgrad, index, dq);
  fe_lc_compute_fed(fe, gamma, q, dq, &value);

  test_assert(fabs(value - 6.1626224e-03) < TEST_FLOAT_TOLERANCE);

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);

  fe_lc_compute_fed(fe, gamma, q, dq, &value);
  test_assert(fabs(value - 6.7087074e-04) < TEST_FLOAT_TOLERANCE);

  /* Dimensionless field 0.2 */
  field[X] = 0.012820969;
  field[Y] = 0.0;
  field[Z] = 0.0;

  fe->param->e0[X] = field[X];
  fe->param->e0[Y] = field[Y];
  fe->param->e0[Z] = field[Z];

  /* Note the electric field now changed so... */

  ic = 1;
  jc = 1;
  kc = 1;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] -  1.2034268e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - -6.0171338e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] -  1.7119419e-02) < TEST_FLOAT_TOLERANCE);

  ic = 2;
  jc = 7;
  kc = 6;
  index = lees_edw_index(le, ic, jc, kc);
  field_tensor(fq, index, q);
  field_grad_tensor_grad(fqgrad, index, dq);
  field_grad_tensor_delsq(fqgrad, index, dsq);

  fe_lc_compute_h(fe, gamma, q, dq, dsq, h);

  test_assert(fabs(h[X][X] - +5.5362629e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Y] -  0.0000000    ) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[X][Z] -  0.0000000    ) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Y] - +5.6693334e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y][Z] - +2.3299416e-03) < TEST_FLOAT_TOLERANCE);

  physics_free(phys);

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

/*****************************************************************************
 *
 *  do_test_fe_lc_device1
 *
 *****************************************************************************/

__host__ int do_test_fe_lc_device1(pe_t * pe, cs_t * cs, fe_lc_t * fe) {

  /* Some parameters */
  double a0 = 0.014384711;
  double gamma = 3.1764706;
  double kappa = 0.01;
  double epsilon = 41.4;
  double redshift = 1.0;
  fe_lc_param_t param = {0};

  dim3 nblk, ntpb;
  physics_t * phys = NULL;
  fe_lc_t * fetarget = NULL;

  assert(pe);
  assert(cs);
  assert(fe);

  physics_create(pe, &phys);

  param.a0 = a0;
  param.gamma = gamma;
  param.kappa0 = kappa;
  param.epsilon = epsilon;
  param.redshift = redshift;
  fe_lc_param_set(fe, &param);
  fe_lc_param_commit(fe);

  fe_lc_target(fe, (fe_t **) &fetarget);

  kernel_launch_param(1, &nblk, &ntpb);
  ntpb.x = 1;

  {
    fe_lc_param_t * p = NULL;
    tdpAssert(tdpMalloc((void **) &p, sizeof(fe_lc_param_t)));
    tdpAssert(tdpMemcpy(p, &param, sizeof(fe_lc_param_t),
			tdpMemcpyHostToDevice));
    tdpLaunchKernel(do_test_fe_lc_kernel1, nblk, ntpb, 0, 0, fetarget, p);
    tdpAssert( tdpDeviceSynchronize() );
    tdpAssert(tdpFree(p));
  }

  physics_free(phys);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_lc_kernel1
 *
 *****************************************************************************/

__global__ void do_test_fe_lc_kernel1(fe_lc_t * fe,
				      const fe_lc_param_t * pref) {

  fe_lc_param_t p;
  PI_DOUBLE(pi);

  assert(fe);

  fe_lc_param(fe, &p);

  /* epsilon is sclaed by a factor of 12pi within fe_lc */

  test_assert(fabs(p.a0 - pref->a0) < DBL_EPSILON);
  test_assert(fabs(p.gamma - pref->gamma) < DBL_EPSILON);
  test_assert(fabs(p.kappa0 - pref->kappa0) < DBL_EPSILON);
  test_assert(fabs(12.0*pi*p.epsilon - pref->epsilon) < FLT_EPSILON);
  test_assert(fabs(p.redshift - pref->redshift) < DBL_EPSILON);

  return;
}

/*****************************************************************************
 *
 *  test_fe_lc_dimensionless_field_strength
 *
 *  Function of only the dielectric anistropy epsilon, gamma, A0, and the
 *  external field.
 *
 *****************************************************************************/

int test_fe_lc_dimensionless_field_strength(pe_t * pe) {

  int ifail = 0;
  PI_DOUBLE(pi);

  assert(pe);

  {
    /* Default must be zero */
    fe_lc_param_t param = {.a0 = 1.0, .gamma = 3.0};
    double ered = -1.0;
    fe_lc_dimensionless_field_strength(&param, &ered);
    assert(fabs(ered - 0.0) < DBL_EPSILON);
    if (fabs(ered - 0.0) < DBL_EPSILON) ifail += 1;
  }

  {
    /* epsilon = 1 in reduced units should give reference value eref */
    double e[3] = {1.0, 1.0, 1.0};
    double a0 = 1.0;
    double gamma = 2.7;
    double epsilon = 1.0/(12.0*pi);
    double eref = sqrt(27.0*3.0/(32.0*pi*a0*gamma));

    fe_lc_param_t param = {.a0 = a0, .gamma = gamma, .epsilon = epsilon,
                           .e0 = {e[X], e[Y], e[Z]}};
    double ered = -1.0;
    fe_lc_dimensionless_field_strength(&param, &ered);
    assert(fabs(ered - eref) < DBL_EPSILON);
    if (fabs(ered - eref) < DBL_EPSILON) ifail += 1;
  }

  return ifail;
}
