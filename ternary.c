/****************************************************************************
 *
 *  ternary.c
 *
 *  Implementation of the surfactant free energy described by
 *  van der Graff and van der Sman [REFERENCE].
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter
 *  [1] \psi is surfactant concentration
 *
 *  The free energy density is:
 *
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "ternary.h"

/* Some values might be
 * a_       = -0.0208333;
 * b_       = +0.0208333;
 * kappa_   = +0.12;
 *
 * kt_      = 0.00056587;
 * epsilon_ = 0.03;
 * beta_    = 0.0;
 * w_       = 0.0;
 */

struct fe_ternary_s {
    fe_t super;                       /* "Superclass" block */
    pe_t * pe;                        /* Parallel environment */
    cs_t * cs;                        /* Coordinate system */
    fe_ternary_param_t * param;         /* Parameters */
    field_t * phi;                    /* Single field with {phi,psi} */
    field_grad_t * dphi;              /* gradients thereof */
    fe_ternary_t * target;              /* Device copy */
};

/* Virtual function table (host) */

static fe_vt_t fe_ternary_hvt = {
    (fe_free_ft)      fe_ternary_free,     /* Virtual destructor */
    (fe_target_ft)    fe_ternary_target,   /* Return target pointer */
    (fe_fed_ft)       fe_ternary_fed,      /* Free energy density */
    (fe_mu_ft)        fe_ternary_mu,       /* Chemical potential */
    (fe_mu_solv_ft)   NULL,
    (fe_str_ft)       fe_ternary_str,      /* Total stress */
    (fe_str_ft)       fe_ternary_str,      /* Symmetric stress */
    (fe_str_ft)       NULL,              /* Antisymmetric stress (not relevant */
    (fe_hvector_ft)   NULL,              /* Not relevant */
    (fe_htensor_ft)   NULL,              /* Not relevant */
    (fe_htensor_v_ft) NULL,              /* Not reelvant */
    (fe_stress_v_ft)  fe_ternary_str_v,    /* Total stress (vectorised version) */
    (fe_stress_v_ft)  fe_ternary_str_v,    /* Symmetric part (vectorised) */
    (fe_stress_v_ft)  NULL               /* Antisymmetric part */
};


static __constant__ fe_ternary_param_t const_param;

/****************************************************************************
 *
 *  fe_ternary_create
 *
 ****************************************************************************/

int fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi,
                    field_grad_t * dphi, fe_ternary_param_t param,
                    fe_ternary_t ** fe) {
    int ndevice;
    fe_ternary_t * obj = NULL;
    
    assert(pe);
    assert(cs);
    assert(fe);
    assert(phi);
    assert(dphi);
    
    obj = (fe_ternary_t *) calloc(1, sizeof(fe_ternary_t));
    assert(obj);
    if (obj == NULL) pe_fatal(pe, "calloc(fe_surf1_t) failed\n");
    
    obj->param = (fe_ternary_param_t *) calloc(1, sizeof(fe_ternary_param_t));
    assert(obj->param);
    if (obj->param == NULL) pe_fatal(pe, "calloc(fe_ternary_param_t) failed\n");
    
    obj->pe = pe;
    obj->cs = cs;
    obj->phi = phi;
    obj->dphi = dphi;
    obj->super.func = &fe_ternary_hvt;
    obj->super.id = FE_TERNARY;
    
    /* Allocate target memory, or alias */
    
    tdpGetDeviceCount(&ndevice);
    
    if (ndevice == 0) {
        fe_ternary_param_set(obj, param);
        obj->target = obj;
    }
    else {
        fe_ternary_param_t * tmp;
        tdpMalloc((void **) &obj->target, sizeof(fe_ternary_t));
        tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
        tdpMemcpy(&obj->target->param, tmp, sizeof(fe_ternary_param_t *),
                  tdpMemcpyHostToDevice);
        /* Now copy. */
        assert(0); /* No implementation */
    }
    
    *fe = obj;
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_free
 *
 ****************************************************************************/

__host__ int fe_ternary_free(fe_ternary_t * fe) {
    
    int ndevice;
    
    assert(fe);
    
    tdpGetDeviceCount(&ndevice);
    if (ndevice > 0) tdpFree(fe->target);
    
    free(fe->param);
    free(fe);
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_ternary_info(fe_ternary_t * fe) {
    
    double sigma, xi0;
    double psi_c;
    pe_t * pe = NULL;
    
    assert(fe);
    
    pe = fe->pe;
    
    fe_surf1_sigma(fe, &sigma1);
    fe_surf1_xi0(fe, &xi1);
    
    pe_info(pe, "Ternary free energy parameters:\n");
    pe_info(pe, "Surface penalty kappa1 = %12.5e\n", fe->param->kappa1);
    pe_info(pe, "Surface penalty kappa2 = %12.5e\n", fe->param->kappa2);
    pe_info(pe, "Surface penalty kappa3 = %12.5e\n", fe->param->kappa3);
    pe_info(pe, "Interface width       = %12.5e\n", fe->param->alpha);
    
    pe_info(pe, "\n");
    pe_info(pe, "Derived quantities\n");
    pe_info(pe, "Interfacial tension   = %12.5e, %12.5e, %12.5e\n", sigma1[0],sigma1[1],sigma1[2]);
    pe_info(pe, "Interfacial width     = %12.5e\n", xi1);
   
    
}

/****************************************************************************
 *
 *  fe_ternary_target
 *
 ****************************************************************************/

__host__ int fe_ternary_target(fe_ternary_t * fe, fe_t ** target) {
    
    assert(fe);
    assert(target);
    
    *target = (fe_t *) fe->target;
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_param_set
 *
 ****************************************************************************/

__host__ int fe_ternary_param_set(fe_ternary_t * fe, fe_ternary_param_t vals) {
    
    assert(fe);
    
    *fe->param = vals;
    
    return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_param
 *
 *****************************************************************************/

__host__ int fe_ternary_param(fe_ternary_t * fe, fe_ternary_param_t * values) {
    assert(fe);
    
    *values = *fe->param;
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_sigma
 *  surface tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__host__ int fe_ternary_sigma(fe_ternary_t * fe,  double * sigma1) {
    
    double alpha, kappa1, kappa2, kappa3;
    
    assert(fe);
    assert(sigma1);assert(sigma1 + 1);assert(sigma1 + 2);
    
    alpha  = fe->param->alpha;
    kappa1 = fe->param->kappa1;
    kappa2 = fe->param->kappa2;
    kappa3 = fe->param->kappa3;
    
    sigma1[0] = alpha*(kappa1 + kappa2)/6;
    
    sigma1[1]= alpha*(kappa2 + kappa3)/6;
    
    sigma1[2]= alpha*(kappa3 + kappa1)/6;
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_xi1
 *
 *  Interfacial width.    interfical width = alpha
 *
 ****************************************************************************/

__host__ int fe_ternary_xi1(fe_ternary_t * fe, double * xi1) {
    
    assert(fe);
    assert(xi1);
    
    *xi1 = fe->param->alpha;
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_fed
 *
 *  This is:
 *   The free energy density is as above.
 *  *   (1/32)kappa1 (\rho + \phi - \psi)^2(2 + \psi - \rho - \phi)^2 + (1/8)kappa1 (\alpha)^2 （\nabla\rho + \nabla\phi - \nalba\psi)^2
 * + (1/32)kappa2 (\rho - \phi - \psi)^2(2 + \psi - \rho + \phi)^2 + (1/8)kappa2 (\alpha)^2（\nabla\rho - \nabla\phi - \nalba\psi)^2
 * + (1/2) kappa3 (\psi)^2(1 - \psi)^2 + (1/2)kappa3 (\alpha)^2 (\nalba\psi)^2
 *
 *
 ****************************************************************************/

__host__ int fe_ternary_fed(fe_ternary_t * fe, int index, double * fed) {
    
    int ia;
    double a, b;
    double field[2];
    double phi;
    double psi;
    double rho;
    double grad[2][3];
    double grho[3];  /* the gradient of rho*/
    
    a = 0.0;
    b = 0.0;
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    
    assert(0);
    phi = field[0];
    psi = field[1];
    
    rho = 1.0;
    grho[3] = 0.0;
    
    for (ia = 0; ia < 3; ia++){
        a + = (grho[ia] + grad[0][ia] - grad[1][ia])*(grho[ia] + grad[0][ia] - grad[1][ia]);
        b + = (grho[ia] - grad[0][ia] - grad[1][ia])*(grho[ia] - grad[0][ia] - grad[1][ia]);
    }
    
    *fed = 0.03125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi)*(2 + psi - rho - phi)*(2 + psi - rho - phi)
    + 0.125*fe->param->kappa1*fe->param->alpha*fe->param->alpha*a
    + 0.03125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi)*(2 + psi - rho + phi)*(2 + psi - rho + phi)
    + 0.125*fe->param->kappa2*fe->param->alpha*fe->param->alpha*b
    + 0.5*fe->param->kappa3*psi*psi*(1 - psi)*(1 - psi)
    + 0.5*fe->param->kappa3*fe->param->alpha*fe->param->alpha*dot_product(grad[1], grad[1]);
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_mu
 *
 Three chemical potentials are present:
 *
 *   \mu_\rho =（1/8）kappa1 (\rho + \phi - \psi)(\rho + \phi - \psi - 2)(\rho + \phi - \psi -1)
 *            - (1/8) kappa2 (\rho - \phi - \psi)(\rho - \phi - \psi - 2)(\rho - \phi - \psi -1)
 *            + (1/4) alpha^2 [(kappa1 + kappa2)(\Delta\psi-\Delta\phi) + (kappa2 - kappa1)(\Delta\rho)]
 *
 *
 *    \mu_\phi =（1/8）kappa1 (\rho + \phi - \psi)(\rho + \phi - \psi - 2)(\rho + \phi - \psi -1)
 *            - (1/8) kappa2 (\rho - \phi - \pasi)(\rho - \phi - \psi - 2)(\rho - \phi - \psi -1)
 *            + (1/4) alpha^2 [(kappa2 - kappa1)(\Delta\rho -\Delta\psi) - (kappa2 + kappa1)(\Delta\phi)]
 *
 *
 *    \mu_\psi = -（1/8）kappa1 (\rho + \phi - \psi)(\rho + \phi - \psi - 2)(\rho + \phi - \psi -1)
 *            - (1/8) kappa2 (\rho - \phi - \psi)(\rho - \phi - \psi - 2)(\rho - \phi - \psi -1)
 *            + kappa3 \psi(\psi - 1)(2 *\psi - 1)
 *            + (1/4) alpha^2 [(kappa1 + kappa2)(\Delta\rho) - (kappa2 - kappa1)(\Delta\phi) - (kappa2 + kappa1 + 4*kappa3)(\Delta\psi)]
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
    assert(mu); assert(mu + 1);assert(mu + 2);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    
    phi = field[0];
    psi = field[1];
    
    delsq_rho = 0.0;
    
    //mu_phi
    mu[0] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi -2)*(rho - phi - psi -1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa2 - fe->param->kappa1)*(delsq_rho - delsq[1])
    - (fe->param->kappa1 + fe->param->kappa2)*delsq[0]);
    
    //mu_psi
    mu[1] = -0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi - 1)
    + fe->param->kappa3*psi*(psi - 1)(2*psi - 1)
    +0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*delsq_rho
    -(fe->param->kappa2 - fe->param->kappa1)*delsq[0]
    - (fe->param->kappa2 + fe->param->kappa1 + 4*fe->param->kappa3)*delsq[1]);
    //mu_rho
    mu[2] = 0.125*fe->param->kappa1*(rho + phi - psi)*(rho + phi - psi - 2)*(rho + phi - psi - 1)
    - 0.125*fe->param->kappa2*(rho - phi - psi)*(rho - phi - psi - 2)*(rho - phi - psi -1)
    + 0.25*fe->param->alpha*fe->param->alpha*((fe->param->kappa1 + fe->param->kappa2)*(delsq[1] - delsq[0])
    + (fe->param->kappa2 - fe->param->kappa1)*delsq_rho);
    
    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *     - kT ln(1 - \psi)
 *     + W \psi \phi^2
 *     + \epsilon \phi \nabla_a \phi \nabla_a \psi
 *     + \epsilon \phi \psi \nabla^2 \phi
 *     + 2 \beta \phi \psi \nabla_a\phi \nabla_a\psi
 *     + \beta\phi\psi^2 \nabla^2 \phi
 *     - (1/2) \beta\psi^2 (\nabla\phi)^2
 *
 *  P_ab = (\kappa - \epsilon\psi - \beta\psi^2) \nabla_a \phi \nabla_b \phi
 *
 ****************************************************************************/

__host__ int fe_ternary_str(fe_ternary_t * fe, int index, double s[3][3]) {
    
    int ia, ib;
    double field[2];
    double phi;
    double psi;
    double rho;
    double delsq[2];
    double grad[2][3];
    double p0;
    KRONECKER_DELTA_CHAR(d);
    
    assert(fe);
    
    field_scalar_array(fe->phi, index, field);
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);
    
    phi = field[0];
    psi = field[1];
    
    p0= (fe->param->kappa1 + fe->param->kappa2)(0.09375*rho*rho*rho*rho + 0.09375*phi*phi*phi*phi + 0.5625*rho*rho*phi*phi
    + 0.5625*rho*rho*psi*psi + 0.5625*phi*phi*psi*psi - 0.375*rho*rho*rho*psi - 0.375*rho*psi*psi*psi + 0.75*rho*rho*psi
    - 0.75*rho*phi*phi - 0.75*rho*psi*psi + 0.75*phi*phi*psi - 0.25*rho*rho*rho + 0.125*rho*rho + 0.125*phi*phi
    - 0.25*rho*psi- 1.125*rho*phi*phi*psi)
    + (fe->param->kappa1 - fe->param->kappa2)(0.375*rho*rho*rho*phi + 0.375*rho*phi*phi*phi - 0.375*phi*phi*phi*psi
    - 0.375*phi*psi*psi*psi -0.25*phi*phi*phi -0.75*rho*rho*phi - 0.75*phi*psi*psi + 0.25*rho*phi - 0.25*phi*psi
    + 1.125*rho*phi*psi*psi - 1.125*rho*rho*phi*psi + 1.5*rho*phi*psi)
    + 0.25*(fe->param->kappa1 + fe->param->kappa2 - 8*fe->param->kappa3)*psi*psi*psi
    +(fe->param->kappa1 + fe->param->kappa2 + 16*fe->param->kappa3)(0.09375*psi*psi*psi*psi + 0.125*psi*psi)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grho,grho)) - rho*delsq_rho)
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(-0.5*(dot_product(grad[0],grad[0])) - phi*delsq[0])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(-0.5*(dot_product(grad[1],grad[1]) - psi*delsq[1])
    + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)(-dot_product(grho,grad[0])-rho*delsq[0]-phi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)(-dot_product(grho,grad[1])-rho*delsq[1]-psi*delsq_rho)
    - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)(-dot_product(grad[0],grad[1])-phi*delsq[1]-psi*delsq[0]);
    
    
    for (ia = 0; ia < 3; ia++) {
        for (ib = 0; ib < 3; ib++) {
            s[ia][ib] = p0*d[ia][ib]
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grho[ib]);
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grad[0][ia]*grad[0][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2 + 4*fe->param->kappa3)*(grad[1][ia]*grad[1][ib])
            + fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grho[ia]*grad[0][ib] + grad[0][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 + fe->param->kappa2)*(grho[ia]*grad[1][ib] + grad[1][ia]*grho[ib])
            - fe->param->alpha*fe->param->alpha*0.25*(fe->param->kappa1 - fe->param->kappa2)*(grad[0][ia]*grad[1][ib] +
                                                                                              grad[1][ia]*grad[0][ib]);
        }
    }
    
    
    return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

int fe_ternary_str_v(fe_ternary_t * fe, int index, double s[3][3][NSIMDVL]) {
    
    int ia, ib;
    int iv;
    double s1[3][3];
    
    assert(fe);
    
    for (iv = 0; iv < NSIMDVL; iv++) {
        fe_ternary_str(fe, index + iv, s1);
        for (ia = 0; ia < 3; ia++) {
            for (ib = 0; ib < 3; ib++) {
                s[ia][ib][iv] = s1[ia][ib];
            }
        }
    }
    
    return 0;
}
