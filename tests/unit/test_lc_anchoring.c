/*****************************************************************************
 *
 *  test_lc_anchoring.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "lc_anchoring.h"

int test_lc_anchoring_type_from_string(void);
int test_lc_anchoring_type_from_enum(void);
int test_lc_anchoring_kappa1_ct(void);
int test_lc_anchoring_fixed_q0(void);
int test_lc_anchoring_fixed_ct(void);
int test_lc_anchoring_normal_q0(void);
int test_lc_anchoring_normal_ct(void);
int test_lc_anchoring_planar_qtilde(void);
int test_lc_anchoring_planar_ct(void);

int test_ref_kappa1_ct(double kappa1, double q0, const double nhat[3],
		       const double qs[3][3], double c[3][3]);
int test_ref_qtperp(double a0, const double qs[3][3], const double nhat[3],
	 	    double qtperp[3][3]);

/*****************************************************************************
 *
 *  test_lc_anchoring_suite
 *
 *****************************************************************************/

int test_lc_anchoring_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lc_anchoring_type_from_string();

  test_lc_anchoring_kappa1_ct();

  test_lc_anchoring_fixed_q0();
  test_lc_anchoring_fixed_ct();
  test_lc_anchoring_normal_q0();
  test_lc_anchoring_normal_ct();
  test_lc_anchoring_planar_qtilde();
  test_lc_anchoring_planar_ct();

  pe_info(pe, "PASS     ./unit/test_lc_anchoring\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_type_from_string
 *
 *****************************************************************************/

int test_lc_anchoring_type_from_string(void) {

  lc_anchoring_enum_t lc = LC_ANCHORING_INVALID;

  lc = lc_anchoring_type_from_string("normal");
  assert(lc == LC_ANCHORING_NORMAL);

  lc = lc_anchoring_type_from_string("planar");
  assert(lc == LC_ANCHORING_PLANAR);

  lc = lc_anchoring_type_from_string("fixed");
  assert(lc == LC_ANCHORING_FIXED);

  lc = lc_anchoring_type_from_string("rubbish");
  assert(lc == LC_ANCHORING_INVALID);

  return (lc != LC_ANCHORING_INVALID);
}

/*****************************************************************************
 *
 *  test_lc_anchoring_type_from_enum
 *
 *****************************************************************************/

int test_lc_anchoring_type_from_enum(void) {

  const char * name = NULL;

  name = lc_anchoring_type_from_enum(LC_ANCHORING_NORMAL);
  assert(strcmp(name, "normal") == 0);

  name = lc_anchoring_type_from_enum(LC_ANCHORING_PLANAR);
  assert(strcmp(name, "planar") == 0);

  name = lc_anchoring_type_from_enum(LC_ANCHORING_FIXED);
  assert(strcmp(name, "fixed") == 0);

  return strcmp(name, "fixed");
}

/*****************************************************************************
 *
 *  test_lc_anchoring_kappa1_ct
 *
 *****************************************************************************/

int test_lc_anchoring_kappa1_ct(void) {

  double kappa1 = 0.01;
  double q0 = 0.02;
  double qs[3][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};

  /* x-normal */
  {
    double nhat[3] = {1.0, 0.0, 0.0};
    double cq[3][3] = {0};
    double cq0[3][3] = {0};

    lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, cq);
    test_ref_kappa1_ct(kappa1, q0, nhat, qs, cq0);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(cq[ia][ib] - cq0[ia][ib]) < DBL_EPSILON);
      }
    }
  }

  /* y-normal */
  {
    double nhat[3] = {0.0, 1.0, 0.0};
    double cq[3][3] = {0};
    double cq0[3][3] = {0};

    lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, cq);
    test_ref_kappa1_ct(kappa1, q0, nhat, qs, cq0);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(cq[ia][ib] - cq0[ia][ib]) < DBL_EPSILON);
      }
    }
  }

  /* z-normal */
  {
    double nhat[3] = {0.0, 0.0, 1.0};
    double cq[3][3] = {0};
    double cq0[3][3] = {0};

    lc_anchoring_kappa1_ct(kappa1, q0, nhat, qs, cq);
    test_ref_kappa1_ct(kappa1, q0, nhat, qs, cq0);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(cq[ia][ib] - cq0[ia][ib]) < DBL_EPSILON);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_fixed_q0
 *
 *****************************************************************************/

int test_lc_anchoring_fixed_q0() {

  double a0 = 2.0;

  double nfix[3]  = {1.0, 2.0, 3.0};
  double q0[3][3] = {0};

  lc_anchoring_fixed_q0(nfix, a0, q0);

  assert(fabs(0.5*a0*(3.0*1.0*nfix[X] - 1.0) - q0[X][X]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*1.0*nfix[Y] - 0.0) - q0[X][Y]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*1.0*nfix[Z] - 0.0) - q0[X][Z]) < DBL_EPSILON);

  assert(fabs(0.5*a0*(3.0*2.0*nfix[X] - 0.0) - q0[Y][X]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*2.0*nfix[Y] - 1.0) - q0[Y][Y]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*2.0*nfix[Z] - 0.0) - q0[Y][Z]) < DBL_EPSILON);

  assert(fabs(0.5*a0*(3.0*3.0*nfix[X] - 0.0) - q0[Z][X]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*3.0*nfix[Y] - 0.0) - q0[Z][Y]) < DBL_EPSILON);
  assert(fabs(0.5*a0*(3.0*3.0*nfix[Z] - 1.0) - q0[Z][Z]) < DBL_EPSILON);

  return 0;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_fixed_ct
 *
 *  Assume nematic (q0 = 0) and just test the surface term in the
 *  energy.
 *
 *****************************************************************************/

int test_lc_anchoring_fixed_ct(void) {

  int ifail = 0;

  double a0 = 2.0;
  double kappa1 = 0.01;
  double q0 = 0.0;
  double qs[3][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
  lc_anchoring_param_t anchor = { .type = LC_ANCHORING_FIXED,
                                  .w1 = 2.0,
				  .w2 = 0.0,
				  .nfix = {1.0, 2.0, 3.0}};
  {
    double nhat[3] = {1.0, 0.0, 0.0};
    double qfix[3][3] = {0};
    double ct[3][3] = {0};

    lc_anchoring_fixed_ct(&anchor, qs, nhat, kappa1, q0, a0, ct);
    lc_anchoring_fixed_q0(anchor.nfix, a0, qfix);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	double ct0 = -anchor.w1*(qs[ia][ib] - qfix[ia][ib]);
	assert(fabs(ct0 - ct[ia][ib]) < DBL_EPSILON);
	if (fabs(ct0 - ct[ia][ib]) > DBL_EPSILON) ifail += 1;
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_normal_q0
 *
 *  Should be the same implementation as lc_anchoring_fixed_q0().
 *
 *****************************************************************************/

int test_lc_anchoring_normal_q0(void) {

  double nhat[3] = {1.0, 2.0, 3.0};
  double a0 = 2.4;
  double q0[3][3] = {0};

  lc_anchoring_normal_q0(nhat, a0, q0);

  assert(fabs(q0[X][X] - 0.5*a0*(3.0*nhat[X]*nhat[X] - 1.0)) < DBL_EPSILON);
  assert(fabs(q0[X][Y] - 0.5*a0*(3.0*nhat[X]*nhat[Y]      )) < DBL_EPSILON);
  assert(fabs(q0[X][Z] - 0.5*a0*(3.0*nhat[X]*nhat[Z]      )) < DBL_EPSILON);
  assert(fabs(q0[Y][X] - 0.5*a0*(3.0*nhat[Y]*nhat[X]      )) < DBL_EPSILON);
  assert(fabs(q0[Y][Y] - 0.5*a0*(3.0*nhat[Y]*nhat[Y] - 1.0)) < DBL_EPSILON);
  assert(fabs(q0[Y][Z] - 0.5*a0*(3.0*nhat[Y]*nhat[Z]      )) < DBL_EPSILON);
  assert(fabs(q0[Z][X] - 0.5*a0*(3.0*nhat[Z]*nhat[X]      )) < DBL_EPSILON);
  assert(fabs(q0[Z][Y] - 0.5*a0*(3.0*nhat[Z]*nhat[Y]      )) < DBL_EPSILON);
  assert(fabs(q0[Z][Z] - 0.5*a0*(3.0*nhat[Z]*nhat[Z] - 1.0)) < DBL_EPSILON);

  return 0;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_normal_ct
 *
 *  Again, as the kappa1 term lc_anchoring_kappa1_ct() is tested
 *  independently, just look at the w1 term here.
 *
 *****************************************************************************/

int test_lc_anchoring_normal_ct(void) {

  int ifail = 0;

  double a0 = 2.0;
  double kappa1 = 0.01;
  double q0 = 0.0;
  double qs[3][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
  lc_anchoring_param_t anchor = { .type = LC_ANCHORING_NORMAL,
                                  .w1 = 2.0,
				  .w2 = 0.0,
				  .nfix = {0}};
  {
    double nhat[3] = {1.0, 1.0, 1.0};
    double qnormal[3][3] = {0};
    double ct[3][3] = {0};

    lc_anchoring_normal_ct(&anchor, qs, nhat, kappa1, q0, a0, ct);
    lc_anchoring_normal_q0(nhat, a0, qnormal);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	double ct0 = -anchor.w1*(qs[ia][ib] - qnormal[ia][ib]);
	assert(fabs(ct0 - ct[ia][ib]) < DBL_EPSILON);
	if (fabs(ct0 - ct[ia][ib]) > DBL_EPSILON) ifail += 1;
      }
    }
  }
  
  return ifail;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_planar_qtilde
 *
 *****************************************************************************/

int test_lc_anchoring_planar_qtilde(void) {

  int ifail = 0;

  double a0 = 3.0;
  double qs[3][3] = {{1.0,2.0,3.0}, {4.0,5.0,6.0}, {7.0,8.0,9.0}};
  double qtilde[3][3] = {0};

  lc_anchoring_planar_qtilde(a0, qs, qtilde);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      double dab = 1.0*(ia == ib);
      double diff = fabs(qtilde[ia][ib] - (qs[ia][ib] + 0.5*a0*dab));
      assert(fabs(diff) < DBL_EPSILON);
      if (fabs(diff) > DBL_EPSILON) ifail += 1;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_planar_ct
 *
 *  Again, set nematic case (kappa1, q0 = 0.0).
 *
 *****************************************************************************/

int test_lc_anchoring_planar_ct(void) {

  int ifail = 0;

  double a0 = 2.0;
  double kappa1 = 0.0;
  double q0 = 0.0;
  double qs[3][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};

  {
    /* w2 = 0 */
    lc_anchoring_param_t anchor = { .type = LC_ANCHORING_PLANAR,
                                    .w1 = 2.0,
				    .w2 = 0.0,
				    .nfix = {0}};
    double nhat[3] = {1.0, 1.0, 1.0};
    double qtilde[3][3] = {0};
    double qtperp[3][3] = {0};
    double ct[3][3] = {0};

    lc_anchoring_planar_ct(&anchor, qs, nhat, kappa1, q0, a0, ct);
    lc_anchoring_planar_qtilde(a0, qs, qtilde);

    test_ref_qtperp(a0, qs, nhat, qtperp);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	double fe = -anchor.w1*(qtilde[ia][ib] - qtperp[ia][ib]);
	assert(fabs(fe - ct[ia][ib]) < DBL_EPSILON);
	if (fabs(fe - ct[ia][ib]) > DBL_EPSILON) ifail += 1;
      }
    }
  }

  {
    /* w1 = 0 */
    lc_anchoring_param_t anchor = { .type = LC_ANCHORING_PLANAR,
                                    .w1 = 0.0,
				    .w2 = 2.0,
				    .nfix = {0}};
    double nhat[3] = {1.0, 1.0, 1.0};
    double qtilde[3][3] = {0};
    double qt2 = 0.0;
    double ct[3][3] = {0};

    lc_anchoring_planar_ct(&anchor, qs, nhat, kappa1, q0, a0, ct);
    lc_anchoring_planar_qtilde(a0, qs, qtilde);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	qt2 += qtilde[ia][ib]*qtilde[ia][ib];
      }
    }

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	double fe = -2.0*anchor.w2*(qt2  - 2.25*a0*a0)*qtilde[ia][ib];
	assert(fabs(fe - ct[ia][ib]) < DBL_EPSILON);
	if (fabs(fe - ct[ia][ib]) > DBL_EPSILON) ifail += 1; 
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_ref_kappa1_ct
 *
 *  A reference calculation for lc_anchoring_kappa1_ct which retains the
 *  permutation tensor.
 *
 *****************************************************************************/

int test_ref_kappa1_ct(double kappa1, double q0, const double nhat[3],
		       const double qs[3][3], double c[3][3]) {

  LEVI_CIVITA_CHAR(epsilon);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {

      c[ia][ib] = 0.0;

      for (int ig = 0; ig < 3; ig++) {
        for (int ih = 0; ih < 3; ih++) {
	  double q_hb = qs[ih][ib];
	  double q_ha = qs[ih][ia];
	  double e_agh = epsilon[ia][ig][ih];
	  double e_bgh = epsilon[ib][ig][ih];
          c[ia][ib] -= kappa1*q0*nhat[ig]*(e_agh*q_hb + e_bgh*q_ha);
        }
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_ref_qtperp
 *
 *****************************************************************************/

int test_ref_qtperp(double a0,
		    const double qs[3][3],
		    const double nhat[3],
		    double qtperp[3][3]) {

  double qtilde[3][3] = {0};

  lc_anchoring_planar_qtilde(a0, qs, qtilde);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      qtperp[ia][ib] = 0.0;
      for (int ig = 0; ig < 3; ig++) {
        for (int ih = 0; ih < 3; ih++) {
          double dag = 1.0*(ia == ig);
          double dhb = 1.0*(ih == ib);
          double pag = dag - nhat[ia]*nhat[ig];
          double phb = dhb - nhat[ih]*nhat[ib];
          qtperp[ia][ib] += pag*qtilde[ig][ih]*phb;
        }
      }
    }
  }

  return 0;
}
