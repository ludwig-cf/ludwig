/*****************************************************************************
 *
 *  test_polar_active.c
 *
 *  Test of the polar active gel free energy against Davide's code.
 *  We check that the free energy density, the moleuclar field, and
 *  the stress are computed correctly for a given order parameter
 *  field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2024 The University of Edinbrugh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

static int test_polar_active_aster(fe_polar_t * fe, cs_t * cs, field_t * fp,
				   field_grad_t * fpgrad);
static int test_polar_active_terms(fe_polar_t * fe, cs_t * cs, field_t * fp,
				   field_grad_t * fpgrad);
static int test_polar_active_init_aster(cs_t * cs, field_t * fp);

/*****************************************************************************
 *
 *  test_polar_active_suite
 *
 *  This is a 2-d test in a system size 100 by 100.
 *
 *****************************************************************************/

int test_polar_active_suite(void) {

  int nf = NVECTOR;
  int nhalo = 2;
  int ntotal[3] = {100, 100, 1};
  int ndevice = 0;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;
  field_t * fp = NULL;
  field_grad_t * fpgrad = NULL;
  fe_polar_t * fe = NULL;

  field_options_t opts = field_options_ndata_nhalo(nf, nhalo);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice > 0 || pe_mpi_size(pe) > 1) {
    pe_info(pe, "SKIP     ./unit/test_polar_active\n");
    pe_free(pe);
    return 0;
  }

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);
  lees_edw_create(pe, cs, NULL, &le);

  field_create(pe, cs, le, "p", &opts, &fp);
  field_grad_create(pe, fp, 2, &fpgrad);
  field_grad_set(fpgrad, grad_2d_5pt_fluid_d2, NULL);

  fe_polar_create(pe, cs, fp, fpgrad, &fe);

  test_polar_active_aster(fe, cs, fp, fpgrad);
  test_polar_active_terms(fe, cs, fp, fpgrad);

  fe_polar_free(fe);
  field_grad_free(fpgrad);
  field_free(fp);

  lees_edw_free(le);
  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_polar_active\n");
  pe_free(pe);

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

static int test_polar_active_aster(fe_polar_t * fe, cs_t * cs, field_t * fp,
				   field_grad_t * fpgrad) {

  int index;

  double fed;
  double p[3];
  double h[3];
  double s[3][3];
  fe_polar_param_t param = {0};

  /* Note that the k2 = 0.02 here is not effective, as all the terms
   * the the polar active are not currently compluted. If all terms
   * were present, the relevant results would be changed. */

  param.a = -0.1;
  param.b = +0.1;
  param.kappa1 = 0.01;
  param.kappa2 = 0.02;
  fe_polar_param_set(fe, param);

  test_polar_active_init_aster(cs, fp);

  /* Order parameter */

  index = cs_index(cs, 1, 1, 1);
  field_vector(fp, index, p);

  test_assert(fabs(p[X] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +7.0710678e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);

  index = cs_index(cs, 2, 28, 1);
  field_vector(fp, index, p);

  test_assert(fabs(p[X] - +9.0523694e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Y] - +4.2490714e-01) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(p[Z]) < TEST_DOUBLE_TOLERANCE);

  /* Gradient terms */

  field_halo_swap(fp, FIELD_HALO_HOST);
  field_grad_compute(fpgrad);

  /* Free energy density (not computed in independent code) */

  index = cs_index(cs, 1, 50, 1);
  fe_polar_fed(fe, index, &fed);

  index = cs_index(cs, 100, 3, 1);
  fe_polar_fed(fe, index, &fed);
  test_assert(fabs(fed - -2.2448448e-02) < TEST_FLOAT_TOLERANCE);

  /* Molecular field */

  index = cs_index(cs, 4, 78, 1);
  fe_polar_mol_field(fe, index, h);
  test_assert(fabs(h[X] - -2.9526261e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - +1.6947361e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);

  index = cs_index(cs, 49, 49, 1);
  fe_polar_mol_field(fe, index, h);
  test_assert(fabs(h[X] - -1.0003585e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Y] - -1.0003585e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(h[Z]) < TEST_DOUBLE_TOLERANCE);

  /* Stress */

  index = cs_index(cs, 3, 90, 1);
  fe_polar_stress(fe, index, s);

  test_assert(fabs(s[X][X] - +1.0398195e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +1.2798462e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +1.2795039e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +1.5748583e-06) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z]) < TEST_DOUBLE_TOLERANCE);

  index = cs_index(cs, 100, 1, 1);
  fe_polar_stress(fe, index, s);

  test_assert(fabs(s[X][X] - +4.8979804e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - -4.9469398e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -5.1509267e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +5.0000000e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z]) < TEST_DOUBLE_TOLERANCE);

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

int test_polar_active_terms(fe_polar_t * fe, cs_t * cs, field_t * fp,
			    field_grad_t * fpgrad) {

  int index;
  int ic, jc, kc;
  int nlocal[3];

  double s[3][3];
  fe_polar_param_t param = {0};

  cs_nlocal(cs, nlocal);

  param.a = -0.1;
  param.b = +0.1;
  param.kappa1 = 0.01;
  param.kappa2 = 0.02;
  param.lambda = 2.1;
  param.zeta   = 0.001;
  fe_polar_param_set(fe, param);

  test_polar_active_init_aster(cs, fp);
  field_halo_swap(fp, FIELD_HALO_HOST);
  field_grad_compute(fpgrad);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	fe_polar_stress(fe, index, s);

	test_assert(fabs(s[Y][Z] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[Z][Y] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[X][Z] - 0.0) < TEST_DOUBLE_TOLERANCE);
	test_assert(fabs(s[Z][X] - 0.0) < TEST_DOUBLE_TOLERANCE);

      }
    }
  }

  index = cs_index(cs, 3, 90, 1);
  fe_polar_stress(fe, index, s);

  /* s_ab(3, 90, 1) */

  test_assert(fabs(s[X][X] - +2.6858170e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - -4.8544429e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - -4.8544463e-04) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - +6.5535744e-05) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - -3.3150277e-04) < TEST_FLOAT_TOLERANCE);

  index = cs_index(cs, 100, 1, 1);
  fe_polar_stress(fe, index, s);

  /* s_ab(100, 1, 1) */
  test_assert(fabs(s[X][X] - -1.5237375e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Y] - +2.0447484e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[X][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][X] - +2.0445444e-02) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Y] - -2.2456775e-03) < TEST_FLOAT_TOLERANCE);
  test_assert(fabs(s[Y][Z]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - +1.3667395e-02) < TEST_FLOAT_TOLERANCE);

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

int test_polar_active_init_aster(cs_t * cs, field_t * fp) {

  int nlocal[3];
  int ic, jc, kc, index;

  double ltot[3];
  double p[3];
  double r;
  double x, y, z, x0, y0, z0;

  cs_nlocal(cs, nlocal);
  cs_ltot(cs, ltot);

  x0 = 0.5*ltot[X];
  y0 = 0.5*ltot[Y];
  z0 = 0.5*ltot[Z];

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
	index = cs_index(cs, ic, jc, kc);
	field_vector_set(fp, index, p);
      }
    }
  }

  return 0;
}
