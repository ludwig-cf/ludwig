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
 *  (c) 2019-2021 The University of Edinburgh
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
#include "map_s.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "fe_ternary.h"

typedef struct solid_s {
  map_t * map;            /* Map structure reference */
  fe_ternary_param_t fe;  /* Free energy parameters (copy) */
  double hrka[3];         /* Wetting gradients phi, psi, rho */
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

__global__ void grad_2d_ternary_solid_kernel(kernel_ctxt_t * ktx,
					     field_grad_t * fg,
					     map_t * map, solid_t fe);


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

  static_solid.hrka[0] =  (-h1/k1 + h2/k2)/a2; /* phi */
  static_solid.hrka[1] =  (-h3/k3        )/a2; /* psi */

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
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  cs_nhalo(fgrad->field->cs, &nextra);
  nextra -= 1;
  cs_nlocal(fgrad->field->cs, nlocal);

  assert(nextra >= 0);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1;          limits.kmax = 1;

  kernel_ctxt_create(fgrad->field->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(grad_2d_ternary_solid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, fgrad->target, static_solid.map->target,
		  static_solid);
  
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/****************************************************************************
 *
 *  grad_2d_ternary_solid_kernel
 *
 ****************************************************************************/

__global__ void grad_2d_ternary_solid_kernel(kernel_ctxt_t * ktx,
					     field_grad_t * fg,
					     map_t * map, solid_t fe) {
  int kindex;
  int kiterations;

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int nf;
    int ic, jc, kc, ic1, jc1;
    int ia, index, p;
    int n;

    int isite[NGRAD_];

    double gradn[3];
    double dphi;
    double delsq;

    int status;

    field_t * phi = NULL;

    nf  = fg->field->nf;
    phi = fg->field;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = 1;

    index = kernel_coords_index(ktx, ic, jc, kc);
    map_status(map, index, &status);
    

    if (status == MAP_FLUID) {

      /* Set solid/fluid flag to index neighbours */

      for (p = 1; p < NGRAD_; p++) {
	ic1 = ic + bs_cv[p][X];
	jc1 = jc + bs_cv[p][Y];

	isite[p] = kernel_coords_index(ktx, ic1, jc1, kc);
	map_status(map, isite[p], &status);
	if (status != MAP_FLUID) isite[p] = -1;
      }

      for (n = 0; n < nf; n++) {

	delsq = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  gradn[ia] = 0.0;
	}
	  
	for (p = 1; p < NGRAD_; p++) {

	  if (isite[p] == -1) {
	    /* Wetting condition */
	    dphi = fe.hrka[n];
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
