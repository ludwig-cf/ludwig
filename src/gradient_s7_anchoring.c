/*****************************************************************************
 *
 *  gradient_s7_anchoring.c
 *
 *  For liquid crystal tensor order parameter Q_ab.
 *
 *  Gradient operations for 3D seven point stencil.
 *
 *                        (ic, jc+1, kc)
 *         (ic-1, jc, kc) (ic, jc  , kc) (ic+1, jc, kc)
 *                        (ic, jc-1, kc)
 *
 *  ...and so in z-direction
 *
 *  d_x Q = [Q(ic+1,jc,kc) - Q(ic-1,jc,kc)] / 2
 *  d_y Q = [Q(ic,jc+1,kc) - Q(ic,jc-1,kc)] / 2
 *  d_z Q = [Q(ic,jc,kc+1) - Q(ic,jc,kc-1)] / 2
 *
 *  nabla^2 Q = Q(ic+1,jc,kc) + Q(ic-1,jc,kc)
 *            + Q(ic,jc+1,kc) + Q(ic,jc-1,kc)
 *            + Q(ic,jc,kc+1) + Q(ic,jc,kc-1) - 6 Q(ic,jc,kc)
 *
 *  The cholesteric anchoring boundary condition specifies the surface
 *  free energy
 *
 *  f_s = w (Q_ab - Q^s_ab)^2
 *
 *  There is a correction related to the surface component of the
 *  molecular field
 *
 *        w (Q_ab - Q^s_ab)
 *
 *  and for cholesterics, one related to the pitch wavenumber q0.
 *
 *  This will also cope with parallel boundaries separated by one fluid
 *  points, whatever the solid involved.
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
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "kernel.h"

#include "gradient_s7_anchoring.h"
#include "lc_anchoring_incl.c"

typedef struct param_s param_t;

struct grad_s7_anch_s {
  pe_t * pe;
  cs_t * cs;
  param_t * param;           /* Boundary condition parameters */
  map_t * map;               /* Supports a map */
  fe_lc_t * fe;              /* Liquid crystal free energy */
  grad_s7_anch_t * target;   /* Device memory */
};

struct param_s {
  double a6inv[3][6];
  double a12inv[3][12][12];
  double a18inv[18][18];
};

static grad_s7_anch_t * static_grad = NULL;
static __constant__ param_t static_param;


__host__ int grad_s7_anchoring_param_init(grad_s7_anch_t * anch);
__host__ int grad_s7_anchoring_param_commit(grad_s7_anch_t * anch);

__global__
void grad_s7_kernel(kernel_ctxt_t * ktx, cs_t * cs, grad_s7_anch_t * anch,
		    fe_lc_t * fe, field_grad_t * fg, map_t * map);
__host__ __device__
int grad_s7_bcs_coeff(double kappa0, double kappa1, const int dn[3],
		      double bc[NSYMM][NSYMM][3]);

__host__ __device__ int grad_s7_boundary_c(fe_lc_param_t * param,
					   grad_s7_anch_t * anch, 
					   const double qs[3][3],
					   const int di[3],
					   double c[3][3]);

/*****************************************************************************
 *
 *  grad_s7_anchoring_create
 *
 *****************************************************************************/

__host__ int grad_s7_anchoring_create(pe_t * pe, cs_t * cs, map_t * map,
				      fe_lc_t * fe, grad_s7_anch_t ** pobj) {

  int ndevice;
  grad_s7_anch_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(fe);

  obj = (grad_s7_anch_t *) calloc(1, sizeof(grad_s7_anch_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(grad_s7_anch_t) failed\n");

  obj->param = (param_t *) calloc(1, sizeof(param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(param_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->map = map;
  obj->fe = fe;
  grad_s7_anchoring_param_init(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    param_t * tmp = NULL;
    tdpMalloc((void **) &obj->target, sizeof(grad_s7_anch_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(static_param));
    tdpMemcpy(&obj->target->param, (const void *) &tmp, sizeof(param_t *),
	      tdpMemcpyHostToDevice);
    grad_s7_anchoring_param_commit(obj);
  }

  static_grad = obj;

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_anchoring_map_set
 *
 *****************************************************************************/

__host__ int grad_s7_anchoring_map_set(map_t * map) {

  assert(map);

  static_grad->map = map;

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_anchoring_free
 *
 ****************************************************************************/

__host__ int grad_s7_anchoring_free(grad_s7_anch_t * grad) {

  assert(grad);

  if (grad->target != grad) tdpFree(grad->target);

  free(grad->param);
  free(grad);

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_archoring_param_commit
 *
 *****************************************************************************/

__host__ int grad_s7_anchoring_param_commit(grad_s7_anch_t * anch) {

  assert(anch);

  tdpMemcpyToSymbol(tdpSymbol(static_param), anch->param, sizeof(param_t), 0,
		    tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_anchoring_d2
 *
 *  Driver routine (a callback).
 *
 *****************************************************************************/

__host__ int grad_s7_anchoring_d2(field_grad_t * fg) {

  int nhalo;
  int nextra;
  int nlocal[3] = {0};
  dim3 nblk, ntpb;
  cs_t * cstarget = NULL;
  kernel_info_t limits = {0};
  kernel_ctxt_t * ctxt = NULL;
  grad_s7_anch_t * anch = NULL; /* get static instance and use */

  assert(static_grad);
  assert(static_grad->fe);

  anch = static_grad;

  cs_nhalo(anch->cs, &nhalo);
  cs_nlocal(anch->cs, nlocal);
  nextra = nhalo - 1;

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(anch->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  grad_s7_anchoring_param_commit(anch);
  fe_lc_param_commit(anch->fe);

  cs_target(anch->cs, &cstarget);

  tdpLaunchKernel(grad_s7_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, cstarget,
		  anch->target, anch->fe->target, fg->target,
		  anch->map->target);
  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_6x6_kernel
 *
 *  This solves the 6x6, 12x12, or 18x18 system by multiplying
 *  the right-hand side by the pre-computed inverse so
 *  x = A^{-1} b
 *
 *****************************************************************************/

__global__
void grad_s7_kernel(kernel_ctxt_t * ktx, cs_t * cs, grad_s7_anch_t * anch,
		    fe_lc_t * fe, field_grad_t * fg, map_t * map) {

  int kindex;
  __shared__ int kiterations;

  kiterations = kernel_iterations(ktx);

  assert(ktx);
  assert(anch);
  assert(fg);
  assert(fg->field);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc, index;
    int str[3];
    int ia, ib, n1, n2;
    int ih, ig;
    int n, nunknown;
    int status[6];
    int normal[3];
    const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    const double bcsign[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    
    double gradn[6][3][2];          /* one-sided partial gradients */
    double dq;
    double qs[3][3];
    double c[3][3];
    
    double bc[6][6][3];
    double b18[18];
    double x18[18];
    double tr;
    const double r3 = (1.0/3.0);

    double kappa0;
    double kappa1;
    field_t * q;

    kappa0 = fe->param->kappa0;
    kappa1 = fe->param->kappa1;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

    cs_strides(cs, str + X, str + Y, str + Z);

    if (map->status[index] == MAP_FLUID) {

      q = fg->field;

      /* Set up partial gradients and identify solid neighbours
       * (unknowns) in various directions. If both neighbours
       * in one coordinate direction are solid, treat as known. */

      nunknown = 0;
      
      for (ia = 0; ia < 3; ia++) {
	
	normal[ia] = ia;
	
	/* Look for outward normals is bcs[] */
	
	ib = 2*ia + 1;
	ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];

	status[2*ia] = map->status[index+ib];	

	ib = 2*ia;
	ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];

	status[2*ia+1] = map->status[index+ib];	

	ig = (status[2*ia    ] != MAP_FLUID);
	ih = (status[2*ia + 1] != MAP_FLUID);
	
	/* Calculate half-gradients assuming they are all knowns */
	
	for (n1 = 0; n1 < NQAB; n1++) {

	  gradn[n1][ia][0] =
	    + q->data[addr_rank1(q->nsites, NQAB, index+str[ia], n1)]
	    - q->data[addr_rank1(q->nsites, NQAB, index,         n1)];
	  gradn[n1][ia][1] =
	    + q->data[addr_rank1(q->nsites, NQAB, index,         n1)]
	    - q->data[addr_rank1(q->nsites, NQAB, index-str[ia], n1)];
	}
	
	gradn[ZZ][ia][0] = -gradn[XX][ia][0] - gradn[YY][ia][0];
	gradn[ZZ][ia][1] = -gradn[XX][ia][1] - gradn[YY][ia][1];
	
	/* Set unknown, with direction, or treat as known (zero grad) */
	
	if (ig + ih == 1) {
	  normal[nunknown] = 2*ia + ih;
	  nunknown += 1;
	}
	else if (ig && ih) {
	  for (n1 = 0; n1 < NSYMM; n1++) {
	    gradn[n1][ia][0] = 0.0;
	    gradn[n1][ia][1] = 0.0;
	  }
	}
	
      }
      

      /* Boundary condition constant terms */
      
      if (nunknown > 0) {
	
	/* Fluid Qab at surface */

	qs[X][X] = q->data[addr_rank1(q->nsites, NQAB, index, XX)];
	qs[X][Y] = q->data[addr_rank1(q->nsites, NQAB, index, XY)];
	qs[X][Z] = q->data[addr_rank1(q->nsites, NQAB, index, XZ)];
	qs[Y][X] = q->data[addr_rank1(q->nsites, NQAB, index, XY)];
	qs[Y][Y] = q->data[addr_rank1(q->nsites, NQAB, index, YY)];
	qs[Y][Z] = q->data[addr_rank1(q->nsites, NQAB, index, YZ)];
	qs[Z][X] = q->data[addr_rank1(q->nsites, NQAB, index, XZ)];
	qs[Z][Y] = q->data[addr_rank1(q->nsites, NQAB, index, YZ)];
	qs[Z][Z] = 0.0 - qs[X][X] - qs[Y][Y];
	
	grad_s7_boundary_c(fe->param, anch, qs, bcs[normal[0]], c);
	
	/* Constant terms all move to RHS (hence -ve sign). Factors
	 * of two in off-diagonals agree with matrix coefficients. */
	
	b18[XX] = -1.0*c[X][X];
	b18[XY] = -2.0*c[X][Y];
	b18[XZ] = -2.0*c[X][Z];
	b18[YY] = -1.0*c[Y][Y];
	b18[YZ] = -2.0*c[Y][Z];
	b18[ZZ] = -1.0*c[Z][Z];
	
	/* Fill in a known value in unknown position so we
	 * and compute a gradient as 0.5*(grad[][][0] + gradn[][][1]) */
	ig = normal[0]/2;
	ih = normal[0]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
      }
      
      if (nunknown > 1) {

	/* Combine the outward normals to produce a unique outward
	 * normal at the edge */
	int bcse[3] = {0, 0, 0};
	int nn0 = normal[0];
	int nn1 = normal[1];
	bcse[X] = bcs[nn0][X] + bcs[nn1][X];
	bcse[Y] = bcs[nn0][Y] + bcs[nn1][Y];
	bcse[Z] = bcs[nn0][Z] + bcs[nn1][Z];

	grad_s7_boundary_c(fe->param, anch, qs, bcse, c);

	/* Overwrite the existing values, and add new ones, which are
	 * the same. */
	b18[0*NSYMM + XX] = -1.0*c[X][X];
	b18[0*NSYMM + XY] = -2.0*c[X][Y];
	b18[0*NSYMM + XZ] = -2.0*c[X][Z];
	b18[0*NSYMM + YY] = -1.0*c[Y][Y];
	b18[0*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[0*NSYMM + ZZ] = -1.0*c[Z][Z];
	b18[1*NSYMM + XX] = -1.0*c[X][X];
	b18[1*NSYMM + XY] = -2.0*c[X][Y];
	b18[1*NSYMM + XZ] = -2.0*c[X][Z];
	b18[1*NSYMM + YY] = -1.0*c[Y][Y];
	b18[1*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[1*NSYMM + ZZ] = -1.0*c[Z][Z];

	/* Set the second known direction partial gradient */
	ig = normal[1]/2;
	ih = normal[1]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
      }

      if (nunknown > 2) {

	int bcse[3] = {0};
	int nn0 = normal[0];
	int nn1 = normal[1];
	int nn2 = normal[2];

	bcse[X] = bcs[nn0][X] + bcs[nn1][X] + bcs[nn2][X];
	bcse[Y] = bcs[nn0][Y] + bcs[nn1][Y] + bcs[nn2][Y];
	bcse[Z] = bcs[nn0][Z] + bcs[nn1][Z] + bcs[nn2][Z];

	grad_s7_boundary_c(fe->param, anch, qs, bcse, c);

	b18[0*NSYMM + XX] = -1.0*c[X][X];
	b18[0*NSYMM + XY] = -2.0*c[X][Y];
	b18[0*NSYMM + XZ] = -2.0*c[X][Z];
	b18[0*NSYMM + YY] = -1.0*c[Y][Y];
	b18[0*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[0*NSYMM + ZZ] = -1.0*c[Z][Z];
	b18[1*NSYMM + XX] = -1.0*c[X][X];
	b18[1*NSYMM + XY] = -2.0*c[X][Y];
	b18[1*NSYMM + XZ] = -2.0*c[X][Z];
	b18[1*NSYMM + YY] = -1.0*c[Y][Y];
	b18[1*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[1*NSYMM + ZZ] = -1.0*c[Z][Z];
	b18[2*NSYMM + XX] = -1.0*c[X][X];
	b18[2*NSYMM + XY] = -2.0*c[X][Y];
	b18[2*NSYMM + XZ] = -2.0*c[X][Z];
	b18[2*NSYMM + YY] = -1.0*c[Y][Y];
	b18[2*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[2*NSYMM + ZZ] = -1.0*c[Z][Z];
	
	ig = normal[2]/2;
	ih = normal[2]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
      }

      
      if (nunknown == 1) {
	
	/* Special case A matrix is diagonal. */
	/* Subtract all three gradient terms from the RHS and then cancel
	 * the one unknown contribution ... works for any normal[0] */
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    for (ia = 0; ia < 3; ia++) {
	      dq = 0.5*(gradn[n2][ia][0] + gradn[n2][ia][1]);
	      b18[n1] -= bc[n1][n2][ia]*dq;
	    }
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[n1] += bc[n1][n2][normal[0]/2]*dq;
	  }
	  
	  b18[n1] *= bcsign[normal[0]];
	  x18[n1] = anch->param->a6inv[normal[0]/2][n1]*b18[n1];
	}
      }
      
      if (nunknown == 2) {
	
	if (normal[0]/2 == X && normal[1]/2 == Y) normal[2] = Z;
	if (normal[0]/2 == X && normal[1]/2 == Z) normal[2] = Y;
	if (normal[0]/2 == Y && normal[1]/2 == Z) normal[2] = X;
	
	/* Compute the RHS for two unknowns and one known */
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;

	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[n1] -= bc[n1][n2][normal[2]]*dq;
	  }
	}
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[NSYMM + n1] -= bc[n1][n2][normal[2]]*dq;
	    
	  }
	}
	
	/* Solve x = A^-1 b depending on unknown conbination */
	/* XY => ia = 0 XZ => ia = 1 YZ => ia = 2 ... */
	
	ia = normal[0]/2 + normal[1]/2 - 1;
	assert(ia == 0 || ia == 1 || ia == 2);
	
	for (n1 = 0; n1 < 2*NSYMM; n1++) {
	  x18[n1] = 0.0;
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    x18[n1] += bcsign[normal[0]]*anch->param->a12inv[ia][n1][n2]*b18[n2];
	  }
	  for (n2 = NSYMM; n2 < 2*NSYMM; n2++) {
	    x18[n1] += bcsign[normal[1]]*anch->param->a12inv[ia][n1][n2]*b18[n2];
	  }
	}
      }
      
      if (nunknown == 3) {
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[n1] *= bcsign[normal[0]];
	}
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[NSYMM + n1] *= bcsign[normal[1]];
	}
	
	grad_s7_bcs_coeff(kappa0, kappa1, bcs[normal[2]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	  }
	  b18[2*NSYMM + n1] *= bcsign[normal[2]];
	}
	
	/* Solve x = A^-1 b */
	
	for (n1 = 0; n1 < 3*NSYMM; n1++) {
	  x18[n1] = 0.0;
	  for (n2 = 0; n2 < 3*NSYMM; n2++) {
	    x18[n1] += anch->param->a18inv[n1][n2]*b18[n2];
	  }
	}
      }
      
      /* Fix the trace (don't store Qzz in the end) */
      
      for (n = 0; n < nunknown; n++) {
	
	tr = r3*(x18[NSYMM*n + XX] + x18[NSYMM*n + YY] + x18[NSYMM*n + ZZ]);
	x18[NSYMM*n + XX] -= tr;
	x18[NSYMM*n + YY] -= tr;
	
	/* Store missing half gradients */
	
	for (n1 = 0; n1 < NQAB; n1++) {
	  gradn[n1][normal[n]/2][normal[n] % 2] = x18[NSYMM*n + n1];
	}
      }
      
      /* The final answer is the sum of partial gradients */

      for (n1 = 0; n1 < NQAB; n1++) {
	fg->delsq[addr_rank1(q->nsites, NQAB, index, n1)] = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  fg->grad[addr_rank2(q->nsites, NQAB, 3, index, n1, ia)] =
	    0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	  fg->delsq[addr_rank1(q->nsites, NQAB, index, n1)]
	    += gradn[n1][ia][0] - gradn[n1][ia][1];
	}
      }

    }
    /* Next site */
  }
 
  return;
}

/*****************************************************************************
 *
 *  grad_s7_anchoring_param_init
 *
 *  Compute and store the inverse matrices used in the boundary conditions.
 *
 *****************************************************************************/

__host__
int grad_s7_anchoring_param_init(grad_s7_anch_t * anch) {

  int ia, n1, n2;
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

  double bc[6][6][3];
  double ** a12inv[3];
  double ** a18inv;
  fe_lc_param_t feparam;
  KRONECKER_DELTA_CHAR(d_);

  assert(anch);
  fe_lc_param(anch->fe, &feparam);

  /* Compute inverse matrices */

  util_matrix_create(12, 12, &(a12inv[0]));
  util_matrix_create(12, 12, &(a12inv[1]));
  util_matrix_create(12, 12, &(a12inv[2]));
  util_matrix_create(18, 18, &a18inv);

  for (ia = 0; ia < 3; ia++) {

    grad_s7_bcs_coeff(feparam.kappa0, feparam.kappa1, bcs[2*ia + 1], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      anch->param->a6inv[ia][n1] = 1.0/bc[n1][n1][ia];
    }

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
	a18inv[ia*NSYMM + n1][0*NSYMM + n2] = 0.5*(1+d_[ia][X])*bc[n1][n2][X];
	a18inv[ia*NSYMM + n1][1*NSYMM + n2] = 0.5*(1+d_[ia][Y])*bc[n1][n2][Y];
	a18inv[ia*NSYMM + n1][2*NSYMM + n2] = 0.5*(1+d_[ia][Z])*bc[n1][n2][Z];

      }
    }
  }

  for (n1 = 0; n1 < 12; n1++) {
    for (n2 = 0; n2 < 12; n2++) {
      a12inv[0][n1][n2] = a18inv[n1][n2];
      a12inv[2][n1][n2] = a18inv[6+n1][6+n2];
    }
  }

  for (n1 = 0; n1 < 6; n1++) {
    for (n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][n2] = a18inv[n1][n2];
      a12inv[1][n1][6+n2] = a18inv[n1][12+n2];
    }
  }

  for (n1 = 6; n1 < 12; n1++) {
    for (n2 = 0; n2 < 6; n2++) {
      a12inv[1][n1][n2] = a18inv[6+n1][n2];
      a12inv[1][n1][6+n2] = a18inv[6+n1][12+n2];
    }
  }

  ia = util_matrix_invert(12, a12inv[0]);
  assert(ia == 0);
  ia = util_matrix_invert(12, a12inv[1]);
  assert(ia == 0);
  ia = util_matrix_invert(12, a12inv[2]);
  assert(ia == 0);
  ia = util_matrix_invert(18, a18inv);
  assert(ia == 0);

  for (n1 = 0; n1 < 18; n1++) {
    for (n2 = 0; n2 < 18; n2++) {
      anch->param->a18inv[n1][n2] = a18inv[n1][n2];
    }  
  }

  for (ia = 0; ia < 3; ia++) {
    for (n1 = 0; n1 < 12; n1++) {
      for (n2 = 0; n2 < 12; n2++) {
	anch->param->a12inv[ia][n1][n2] = a12inv[ia][n1][n2];
      }
    }
  }

  util_matrix_free(18, &a18inv);
  util_matrix_free(12, &(a12inv[2]));
  util_matrix_free(12, &(a12inv[1]));
  util_matrix_free(12, &(a12inv[0]));

  return 0;
}


/*****************************************************************************
 *
 *  grad_s7_bcs_coeff
 *
 *  Full set of coefficients in boundary condition equation for given
 *  surface normal dn.
 *
 *****************************************************************************/

__host__ __device__
int grad_s7_bcs_coeff(double kappa0, double kappa1, const int dn[3],
		      double bc[NSYMM][NSYMM][3]) {
  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X];
  bc[XX][XY][X] = -kappa1*dn[Y];
  bc[XX][XZ][X] = -kappa1*dn[Z];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;
  bc[XX][ZZ][X] =  0.0;

  bc[XX][XX][Y] =  kappa1*dn[Y];
  bc[XX][XY][Y] =  kappa0*dn[X];
  bc[XX][XZ][Y] =  0.0;
  bc[XX][YY][Y] =  0.0;
  bc[XX][YZ][Y] =  0.0;
  bc[XX][ZZ][Y] =  0.0;

  bc[XX][XX][Z] =  kappa1*dn[Z];
  bc[XX][XY][Z] =  0.0;
  bc[XX][XZ][Z] =  kappa0*dn[X];
  bc[XX][YY][Z] =  0.0;
  bc[XX][YZ][Z] =  0.0;
  bc[XX][ZZ][Z] =  0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y];
  bc[XY][XY][X] =  kappa2*dn[X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y];
  bc[XY][YZ][X] = -kappa1*dn[Z];
  bc[XY][ZZ][X] =  0.0;

  bc[XY][XX][Y] = -kappa1*dn[X];
  bc[XY][XY][Y] =  kappa2*dn[Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z];
  bc[XY][YY][Y] =  kappa0*dn[X];
  bc[XY][YZ][Y] =  0.0;
  bc[XY][ZZ][Y] =  0.0;

  bc[XY][XX][Z] =  0.0;
  bc[XY][XY][Z] =  2.0*kappa1*dn[Z];
  bc[XY][XZ][Z] =  kappa0*dn[Y];
  bc[XY][YY][Z] =  0.0;
  bc[XY][YZ][Z] =  kappa0*dn[X];
  bc[XY][ZZ][Z] =  0.0;

  /* XZ equation */

  bc[XZ][XX][X] =  kappa0*dn[Z];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X];
  bc[XZ][YY][X] =  0.0;
  bc[XZ][YZ][X] = -kappa1*dn[Y];
  bc[XZ][ZZ][X] = -kappa1*dn[Z];

  bc[XZ][XX][Y] =  0.0;
  bc[XZ][XY][Y] =  kappa0*dn[Z];
  bc[XZ][XZ][Y] =  2.0*kappa1*dn[Y];
  bc[XZ][YY][Y] =  0.0;
  bc[XZ][YZ][Y] =  kappa0*dn[X];
  bc[XZ][ZZ][Y] =  0.0;

  bc[XZ][XX][Z] = -kappa1*dn[X];
  bc[XZ][XY][Z] = -kappa1*dn[Y];
  bc[XZ][XZ][Z] =  kappa2*dn[Z];
  bc[XZ][YY][Z] =  0.0;
  bc[XZ][YZ][Z] =  0.0;
  bc[XZ][ZZ][Z] =  kappa0*dn[X];

  /* YY equation */

  bc[YY][XX][X] =  0.0;
  bc[YY][XY][X] =  kappa0*dn[Y];
  bc[YY][XZ][X] =  0.0;
  bc[YY][YY][X] =  kappa1*dn[X];
  bc[YY][YZ][X] =  0.0;
  bc[YY][ZZ][X] =  0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z];
  bc[YY][ZZ][Y] =  0.0;

  bc[YY][XX][Z] =  0.0;
  bc[YY][XY][Z] =  0.0;
  bc[YY][XZ][Z] =  0.0;
  bc[YY][YY][Z] =  kappa1*dn[Z];
  bc[YY][YZ][Z] =  kappa0*dn[Y];
  bc[YY][ZZ][Z] =  0.0;

  /* YZ equation */

  bc[YZ][XX][X] =  0.0;
  bc[YZ][XY][X] =  kappa0*dn[Z];
  bc[YZ][XZ][X] =  kappa0*dn[Y];
  bc[YZ][YY][X] =  0.0;
  bc[YZ][YZ][X] =  2.0*kappa1*dn[X];
  bc[YZ][ZZ][X] =  0.0;

  bc[YZ][XX][Y] =  0.0;
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X];
  bc[YZ][YY][Y] =  kappa0*dn[Z];
  bc[YZ][YZ][Y] =  kappa2*dn[Y];
  bc[YZ][ZZ][Y] = -kappa1*dn[Z];

  bc[YZ][XX][Z] =  0.0;
  bc[YZ][XY][Z] = -kappa1*dn[X];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa1*dn[Y];
  bc[YZ][YZ][Z] =  kappa2*dn[Z];
  bc[YZ][ZZ][Z] =  kappa0*dn[Y];

  /* ZZ equation */

  bc[ZZ][XX][X] =  0.0;
  bc[ZZ][XY][X] =  0.0;
  bc[ZZ][XZ][X] =  kappa0*dn[Z];
  bc[ZZ][YY][X] =  0.0;
  bc[ZZ][YZ][X] =  0.0;
  bc[ZZ][ZZ][X] =  kappa1*dn[X];
  
  bc[ZZ][XX][Y] =  0.0;
  bc[ZZ][XY][Y] =  0.0;
  bc[ZZ][XZ][Y] =  0.0;
  bc[ZZ][YY][Y] =  0.0;
  bc[ZZ][YZ][Y] =  kappa0*dn[Z];
  bc[ZZ][ZZ][Y] =  kappa1*dn[Y];
  
  bc[ZZ][XX][Z] =  0.0;
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X];
  bc[ZZ][YY][Z] =  0.0;
  bc[ZZ][YZ][Z] = -kappa1*dn[Y];
  bc[ZZ][ZZ][Z] =  kappa0*dn[Z];

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_boundary_wall
 *
 *  Compute the constant term in the cholesteric boundary condition.
 *  Fluid point is (ic, jc, kc) with fluid Q_ab = qs
 *
 *  The outward normal is di[3] = {+/-1,+/-1,+/-1}.
 *
 *****************************************************************************/

__host__ __device__ void grad_s7_boundary_wall(fe_lc_param_t * param,
					       const double qs[3][3],
					       const int di[3],
					       double c[3][3]) {

  double nhat[3] = {1.0*di[X], 1.0*di[Y], 1.0*di[Z]};
  double amp     = 0.0;
  double kappa1  = param->kappa1;
  double q0      = param->q0;

  fe_lc_amplitude_compute(param, &amp);

  /* Make sure we have a unit vector */
  {
    double rd = 1.0/sqrt(nhat[X]*nhat[X] + nhat[Y]*nhat[Y] + nhat[Z]*nhat[Z]);
    nhat[X] *= rd;
    nhat[Y] *= rd;
    nhat[Z] *= rd;
  }

  {
    lc_anchoring_param_t * anchor = &param->wall;

    if (anchor->type == LC_ANCHORING_FIXED) {
      lc_anchoring_fixed_ct(anchor, qs, nhat, kappa1, q0, amp, c);
    }

    if (anchor->type == LC_ANCHORING_NORMAL) {
      lc_anchoring_normal_ct(anchor, qs, nhat, kappa1, q0, amp, c);
    }

    if (anchor->type == LC_ANCHORING_PLANAR) {
      lc_anchoring_planar_ct(anchor, qs, nhat, kappa1, q0, amp, c);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  NEED TO switch for colloid/wall cf old version.
 *
 *****************************************************************************/

__host__ __device__ int grad_s7_boundary_c(fe_lc_param_t * param,
					   grad_s7_anch_t * anch,
					   const double qs[3][3],
					   const int di[3],
					   double c[3][3]) {

  grad_s7_boundary_wall(param, qs, di, c);

  return 0;
}
