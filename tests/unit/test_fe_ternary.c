/*****************************************************************************
 *
 *  test_fe_tenery.c
 *
 *  Unit tests for ternary free energy.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Shan Chen (shan.chen@epfl.ch)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "fe_ternary.h"
#include "tests.h"

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi);

/*****************************************************************************
 *
 *  test_fe_tenrary_suite
 *
 *****************************************************************************/

__host__ int test_fe_ternary_suite(void) {

  const int nf2 = 2;
  const int nhalo = 2;

  int ndevice;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  field_t * phi = NULL;

  tdpGetDeviceCount(&ndevice);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_fe_ternary\n");
  }
  else {

    cs_create(pe, &cs);
    cs_init(cs);

    field_create(pe, cs, nf2, "ternary", &phi);
    field_init(phi, nhalo, NULL);

    test_fe_ternary_create(pe, cs, phi);
    /*
    test_fe_surf_create(pe, cs, phi);
    test_fe_surf_xi_etc(pe, cs, phi);
    test_fe_surf_fed(pe, cs, phi);
    test_fe_surf_mu(pe, cs, phi);
    test_fe_surf_str(pe, cs, phi);
    */

    field_free(phi);
    cs_free(cs);
  }

  pe_info(pe, "PASS     ./unit/test_fe_ternary\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_ternary_create
 *
 *****************************************************************************/

__host__ int test_fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_ternary_t * fe = NULL;
  fe_ternary_param_t pref;
  field_grad_t * dphi = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_ternary_create(pe, cs, phi, dphi, pref, &fe);

  assert(fe);

  fe_ternary_free(fe);
  field_grad_free(dphi);

  return 0;
}

#ifdef TO_BE_REFERENCE

/****************************************************************************
 *
 *  fe_ternary_fed
 *
 ****************************************************************************/

__host__ int fe_ternary_fed(fe_ternary_t * fe, int index, double * fed) {
    
    int ia,ic;
    double a[3], b[3];
    double field[2];
    double phi;
    double psi;
    double rho;
    double grad[2][3];
    double grho[3];  /* the gradient of rho*/
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    
    assert(0);
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    rho = 1.0;
    
    for (ic=0;ic<3; ic++){
        grho[ic] = 0.0;
        a[ic] = 0.0;
        b[ic] = 0.0;
    }
    
    for (ia = 0; ia < 3; ia++){
        a[ia] = grho[ia] + grad[0][ia] - grad[1][ia];
        b[ia] = grho[ia] - grad[0][ia] - grad[1][ia];
    }
    
    *fed = 0.03125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi)*(2 + psi - rho - phi)*(2 + psi - rho - phi)
    + 0.125*fe->param->kappa1*fe->param->alpha*fe->param->alpha*dot_product(a,a)
    + 0.03125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi)*(2 + psi - rho + phi)*(2 + psi - rho + phi)
    + 0.125*fe->param->kappa2*fe->param->alpha*fe->param->alpha*dot_product(b,b)
    + 0.5*fe->param->kappa3*psi*psi*(1 - psi)*(1 - psi)
    + 0.5*fe->param->kappa3*fe->param->alpha*fe->param->alpha*dot_product(grad[1], grad[1]);
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_mu
 *
 ****************************************************************************/

__host__ int fe_ternary_mu(fe_ternary_t * fe, int index, double * mu) {
    
    double phi;
    double psi;
    double rho;
    double field[2];
    double delsq[2];
    double delsq_rho; /* laplace operator*/
    
    assert(fe);
    assert(mu); assert(mu + 1);//assert(mu + 2);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    rho = 1.0;
    delsq_rho = 0.0;
    
    //mu_phi
    mu[0] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi - 1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa2 - fe->param->kappa1)*(delsq_rho - delsq[1])
    - (fe->param->kappa1 + fe->param->kappa2)*delsq[0]);
    
    //mu_psi
    mu[1] = -0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi - 1)
    + fe->param->kappa3*psi*(psi - 1)*(2*psi - 1)
    +0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*delsq_rho
    -(fe->param->kappa2 - fe->param->kappa1)*delsq[0]
    - (fe->param->kappa2 + fe->param->kappa1 + 4*fe->param->kappa3)*delsq[1]);
    //mu_rho
  /*  mu[2] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi -1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*(delsq[1] - delsq[0])
    + (fe->param->kappa2 - fe->param->kappa1)*delsq_rho);*/
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 ****************************************************************************/

__host__ int fe_ternary_str(fe_ternary_t * fe, int index, double s[3][3]) {
    
    int ia, ib,ic;
    double field[2];
    double phi;
    double psi;
    double rho;
    double delsq[2];
    double grad[2][3];
    double p0;
    double grho[3];
    double delsq_rho;
    KRONECKER_DELTA_CHAR(d);
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    rho = 1.0;
    phi = field[0];
    psi = field[1];
    assert(abs(phi)+abs(psi) == 1);
    
    for (ic=0;ic<3; ic++){
        grho[ic] = 0.0;
    }
    
    delsq_rho = 0.0;
    
   
    
    p0= (fe->param->kappa1 + fe->param->kappa2)*(0.09375*rho*rho*rho*rho + 0.09375*phi*phi*phi*phi + 0.5625*rho*rho*phi*phi
    + 0.5625*rho*rho*psi*psi + 0.5625*phi*phi*psi*psi - 0.375*rho*rho*rho*psi - 0.375*rho*psi*psi*psi + 0.75*rho*rho*psi
    - 0.75*rho*phi*phi - 0.75*rho*psi*psi + 0.75*phi*phi*psi - 0.25*rho*rho*rho + 0.125*rho*rho + 0.125*phi*phi
    - 0.25*rho*psi- 1.125*rho*phi*phi*psi)
    + (fe->param->kappa1 - fe->param->kappa2)*(0.375*rho*rho*rho*phi + 0.375*rho*phi*phi*phi - 0.375*phi*phi*phi*psi
    - 0.375*phi*psi*psi*psi -0.25*phi*phi*phi -0.75*rho*rho*phi - 0.75*phi*psi*psi + 0.25*rho*phi - 0.25*phi*psi
    + 1.125*rho*phi*psi*psi - 1.125*rho*rho*phi*psi + 1.5*rho*phi*psi)
    + 0.25*(fe->param->kappa1 + fe->param->kappa2 - 8*fe->param->kappa3)*psi*psi*psi
    +(fe->param->kappa1 + fe->param->kappa2 + 16*fe->param->kappa3)*(0.09375*psi*psi*psi*psi + 0.125*psi*psi)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grho,grho)) - rho*delsq_rho)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grad[0],grad[0])) - phi*delsq[0])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(-0.5*(dot_product(grad[1],grad[1])) - psi*delsq[1])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(-dot_product(grho,grad[0])-rho*delsq[0]-phi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-dot_product(grho,grad[1])-rho*delsq[1]-psi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(-dot_product(grad[0],grad[1])-phi*delsq[1]-psi*delsq[0]);
    
    
    for (ia = 0; ia < 3; ia++) {
        for (ib = 0; ib < 3; ib++) {
            s[ia][ib] = p0*d[ia][ib] + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grho[ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grad[0][ia]*grad[0][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(grad[1][ia]*grad[1][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grho[ia]*grad[0][ib] + grad[0][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grad[1][ib] + grad[1][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grad[0][ia]*grad[1][ib] + grad[1][ia]*grad[0][ib]);
        }
    }
        
    return 0;
}
#endif
