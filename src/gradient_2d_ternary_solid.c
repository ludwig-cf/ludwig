/*****************************************************************************
 *
 *  gradient_2d_ternary_solid.c
 *
 *  Wetting condition for ternary free energy following Semprebon et al.
 *  We assume there is a single solid free energy with associated
 *  constants h_1 and h_2.
 *
 *  No Lees-Edwards planes are supported.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (chan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "gradient_2d_ternary_solid.h"

typedef struct wetting_s {
  double hrka[3];         /* Wetting gradients phi, psi, rho */
} wetting_t;

typedef struct solid_s {
  map_t * map;            /* Map structure reference */
  fe_ternary_param_t fe;  /* Free energy parameters (copy) */
  wetting_t wetting;      /* Wetting parameters */
} solid_t;

static solid_t static_solid = {0};

/* These are the 'links' used to form the gradients at boundaries. */

#define NGRAD_ 9
static __constant__ int bs_cv[NGRAD_][2] = {{ 0, 0},
				 {-1,-1}, {-1, 0}, {-1, 1},
                                 { 0,-1}, { 0, 1}, { 1,-1},
				 { 1, 0}, { 1, 1}};
#define w0 (16.0/36.0)
#define w1  (4.0/36.0)
#define w2  (1.0/36.0)

static __constant__ double wv[NGRAD_] = {w0, w2, w1, w2, w1, w1, w2, w1, w2};

__global__ void grad_2d_ternary_solid_kernel(kernel_3d_t k3d,
					     field_grad_t * fg,
					     map_t * map,
					     wetting_t wet);

/*****************************************************************************
 *
 *  grad_2d_ternary_solid_set
 *
 *****************************************************************************/

__host__ int grad_2d_ternary_solid_set(map_t * map) {

  assert(map);

  static_solid.map = map;

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_ternary_solid_fe_set
 *
 *  Keep a copy of the free energy paramerers.
 *  Set wetting parameters for phi and psi. See Eq. 24-26 Semprebon.
 *
 *  The constraint Eq. 32 that only h1 and h2 are independent may be
 *  imposed elsewhere.
 *
 *****************************************************************************/

__host__ int grad_2d_ternary_solid_fe_set(fe_ternary_t * fe) {


  double k1, k2, k3;
  double h1, h2, h3;
  double a2;

  assert(fe);

  static_solid.fe = *fe->param;

  h1 = fe->param->h1;
  h2 = fe->param->h2;
  h3 = fe->param->h3;
  k1 = fe->param->kappa1;
  k2 = fe->param->kappa2;
  k3 = fe->param->kappa3;
  a2 = fe->param->alpha*fe->param->alpha;

  static_solid.wetting.hrka[0] =  (-h1/k1 + h2/k2)/a2; /* phi */
  static_solid.wetting.hrka[1] =  (-h3/k3        )/a2; /* psi */
  static_solid.wetting.hrka[2] = 0.0;                  /* not used */

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_ternary_solid_d2
 *
 *****************************************************************************/

__host__ int grad_2d_ternary_solid_d2(field_grad_t * fgrad) {

  int nextra;
  int nlocal[3];

  cs_nhalo(fgrad->field->cs, &nextra);
  nextra -= 1;
  cs_nlocal(fgrad->field->cs, nlocal);

  assert(nextra >= 0);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {
      .imin = 1 - nextra, .imax = nlocal[X] + nextra,
      .jmin = 1 - nextra, .jmax = nlocal[Y] + nextra,
      .kmin = 1,          .kmax = 1
    };
    kernel_3d_t k3d = kernel_3d(fgrad->field->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(grad_2d_ternary_solid_kernel, nblk, ntpb, 0, 0,
		    k3d, fgrad->target, static_solid.map->target,
		    static_solid.wetting);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/****************************************************************************
 *
 *  grad_2d_ternary_solid_kernel
 *
 ****************************************************************************/

__global__ void grad_2d_ternary_solid_kernel(kernel_3d_t k3d,
					     field_grad_t * fg,
					     map_t * map,
					     wetting_t wet) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d. kiterations, 1) {

    int nf;
    int ic, jc, kc, ic1, jc1;
    int ia, index, p;

    int isite[NGRAD_];

    double gradn[3];
    double dphi;
    double delsq;

    int status;

    field_t * phi = NULL;

    nf  = fg->field->nf;
    phi = fg->field;

    ic = kernel_3d_ic(&k3d, kindex);
    jc = kernel_3d_jc(&k3d, kindex);
    kc = 1;

    index = kernel_3d_cs_index(&k3d, ic, jc, kc);
    map_status(map, index, &status);


    if (status == MAP_FLUID) {

      /* Set solid/fluid flag to index neighbours */

      for (p = 1; p < NGRAD_; p++) {
	ic1 = ic + bs_cv[p][X];
	jc1 = jc + bs_cv[p][Y];

	isite[p] = kernel_3d_cs_index(&k3d, ic1, jc1, kc);
	map_status(map, isite[p], &status);
	if (status != MAP_FLUID) isite[p] = -1;
      }

      for (int n = 0; n < nf; n++) {

	delsq = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  gradn[ia] = 0.0;
	}

	for (p = 1; p < NGRAD_; p++) {

	  if (isite[p] == -1) {
	    /* Wetting condition */
	    dphi = wet.hrka[n];
	  }
	  else {
	    /* Fluid */
	    dphi = phi->data[addr_rank1(phi->nsites, nf, isite[p], n)]
	         - phi->data[addr_rank1(phi->nsites, nf, index,    n)];
	  }

	  gradn[X] += 3.0*wv[p]*bs_cv[p][X]*dphi;
	  gradn[Y] += 3.0*wv[p]*bs_cv[p][Y]*dphi;
	  delsq    += 6.0*wv[p]*dphi;
	}

	/* Accumulate the final gradients */

	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,X)] = gradn[X];
	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,Y)] = gradn[Y];
	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,Z)] = 0.0;
	fg->delsq[addr_rank1(phi->nsites,nf, index,n)  ] = delsq;
      }

      /* Next fluid site */
    }
  }

  return;
}
