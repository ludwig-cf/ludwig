/*****************************************************************************
 *
 *  gradient_3d_7pt_solid.c
 *
 *  Liquid Crystal tensor order parameter Q_ab.
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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
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
#include "colloids_s.h"
#include "map_s.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "gradient_3d_7pt_solid.h"

typedef struct param_s param_t;

struct grad_lc_anch_s {
  param_t * param;           /* Boundary condition parameters */
  map_t * map;               /* Supports a map */
  colloids_info_t * cinfo;   /* Supports colloids */
  fe_lc_t * fe;              /* Liquid crystal free energy */
  grad_lc_anch_t * target;   /* Device memory */
};

struct param_s {
  double a6inv[3][6];
  double a12inv[3][12][12];
  double a18inv[18][18];
};

static grad_lc_anch_t * static_grad = NULL;
static __constant__ param_t static_param;


__host__ int gradient_param_commit(grad_lc_anch_t * anch);
__host__ int gradient_6x6(grad_lc_anch_t * anch, field_grad_t *grad,
			  int nextra);
__global__
void gradient_6x6_kernel(kernel_ctxt_t * ktx, grad_lc_anch_t * anch,
			 fe_lc_t * fe, field_grad_t * fg,
			 map_t * map, colloids_info_t * cinfo);
__host__ __device__
int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
			  double bc[NSYMM][NSYMM][3]);
__host__ __device__
int colloids_q_boundary(fe_lc_param_t * param, const double n[3],
			double qs[3][3], double q0[3][3],
			int map_status);
__host__ __device__
int q_boundary_constants(fe_lc_param_t * param, int ic, int jc, int kc,
			 double qs[3][3],
			 const int di[3], int status, double c[3][3],
			 colloids_info_t * cinfo);

/*****************************************************************************
 *
 *  grad_lc_anch_create
 *
 *****************************************************************************/

__host__ int grad_lc_anch_create(map_t * map, colloids_info_t * cinfo,
				 fe_lc_t * fe, grad_lc_anch_t ** pobj) {

  int ndevice;
  grad_lc_anch_t * obj = NULL;

  assert(fe);

  obj = (grad_lc_anch_t *) calloc(1, sizeof(grad_lc_anch_t));
  if (obj == NULL) fatal("calloc(grad_lc_anch_t) failed\n");

  obj->param = (param_t *) calloc(1, sizeof(param_t));
  if (obj->param == NULL) fatal("calloc(param_t) failed\n");

  obj->map = map;
  obj->cinfo = cinfo;
  obj->fe = fe;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    assert(0);
  }

  static_grad = obj;

  return 0;
}

/*****************************************************************************
 *
 *  grad_lc_anch_free
 *
 ****************************************************************************/

__host__ int grad_lc_anch_free(grad_lc_anch_t * grad) {

  assert(grad);

  if (grad->target != grad) targetFree(grad->target);

  free(grad->param);
  free(grad);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_solid_set
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_solid_set(map_t * map, colloids_info_t * cinfo) {

  assert(map);
  assert(static_grad);

  static_grad->map = map;
  static_grad->cinfo = cinfo;

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_solid_d2
 *
 *  Driver routine (a callback).
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_solid_d2(field_grad_t * fg) {

  int nextra;
  grad_lc_anch_t * grad = NULL; /* get static instance and use */

  assert(static_grad);
  assert(static_grad->fe);

  grad = static_grad;

  nextra = coords_nhalo() - 1;

  gradient_6x6(grad, fg, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_6x6
 *
 *  This solves the boundary condition equation by pre-computing
 *  the inverse of the system matrix for a number of cases.
 *
 *  Written for GPU, but will work anywhere.
 *
 *****************************************************************************/

__host__
int gradient_6x6(grad_lc_anch_t * anch, field_grad_t * fg, int nextra) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(anch);
  assert(fg);

  coords_nlocal(nlocal);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  gradient_param_commit(anch);
  fe_lc_param_commit(anch->fe);

  __host_launch(gradient_6x6_kernel, nblk, ntpb, ctxt->target,
		anch->target, anch->fe->target, fg->target,
		anch->map->target, anch->cinfo->tcopy);

  targetDeviceSynchronise();

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
void gradient_6x6_kernel(kernel_ctxt_t * ktx, grad_lc_anch_t * anch,
			 fe_lc_t * fe, field_grad_t * fg,
			 map_t * map, colloids_info_t * cinfo) {

  int kindex;
  __shared__ int kiterations;

  kiterations = kernel_iterations(ktx);

  assert(ktx);
  assert(anch);
  assert(fg);

  __target_simt_parallel_for(kindex, kiterations, 1) {

    int ic, jc, kc, index;
    int str[3];
    int ia, ib, n1, n2;
    int ih, ig;
    int n, nunknown;
    int nsites;
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

    kappa0 = fe->param->kappa0;
    kappa1 = fe->param->kappa1;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);

#ifdef __NVCC__
    assert(0);
#else
    coords_strides(str + X, str + Y, str + Z);
#endif

    if (map->status[index] == MAP_FLUID) {

      nsites = fg->field->nsites;

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
	    + fg->field->data[addr_rank1(nsites, NQAB, index+str[ia], n1)]
	    - fg->field->data[addr_rank1(nsites, NQAB, index,         n1)];
	  gradn[n1][ia][1] =
	    + fg->field->data[addr_rank1(nsites, NQAB, index,         n1)]
	    - fg->field->data[addr_rank1(nsites, NQAB, index-str[ia], n1)];
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

	qs[X][X] = fg->field->data[addr_rank1(nsites, NQAB, index, XX)];
	qs[X][Y] = fg->field->data[addr_rank1(nsites, NQAB, index, XY)];
	qs[X][Z] = fg->field->data[addr_rank1(nsites, NQAB, index, XZ)];
	qs[Y][X] = fg->field->data[addr_rank1(nsites, NQAB, index, XY)];
	qs[Y][Y] = fg->field->data[addr_rank1(nsites, NQAB, index, YY)];
	qs[Y][Z] = fg->field->data[addr_rank1(nsites, NQAB, index, YZ)];
	qs[Z][X] = fg->field->data[addr_rank1(nsites, NQAB, index, XZ)];
	qs[Z][Y] = fg->field->data[addr_rank1(nsites, NQAB, index, YZ)];
	qs[Z][Z] = 0.0
	  - fg->field->data[addr_rank1(nsites, NQAB, index, XX)]
	  - fg->field->data[addr_rank1(nsites, NQAB, index, YY)];
	
	q_boundary_constants(fe->param, ic, jc, kc, qs, bcs[normal[0]],
			     status[normal[0]], c, cinfo);
	
	/* Constant terms all move to RHS (hence -ve sign). Factors
	 * of two in off-diagonals agree with matrix coefficients. */
	
	b18[XX] = -1.0*c[X][X];
	b18[XY] = -2.0*c[X][Y];
	b18[XZ] = -2.0*c[X][Z];
	b18[YY] = -1.0*c[Y][Y];
	b18[YZ] = -2.0*c[Y][Z];
	b18[ZZ] = -1.0*c[Z][Z];
	
	/* Fill a a known value in unknown position so we
	 * and compute a gradient as 0.5*(grad[][][0] + gradn[][][1]) */
	ig = normal[0]/2;
	ih = normal[0]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
      }
      
      if (nunknown > 1) {
	
	q_boundary_constants(fe->param, ic, jc, kc, qs, bcs[normal[1]],
			     status[normal[1]], c, cinfo);
	
	b18[1*NSYMM + XX] = -1.0*c[X][X];
	b18[1*NSYMM + XY] = -2.0*c[X][Y];
	b18[1*NSYMM + XZ] = -2.0*c[X][Z];
	b18[1*NSYMM + YY] = -1.0*c[Y][Y];
	b18[1*NSYMM + YZ] = -2.0*c[Y][Z];
	b18[1*NSYMM + ZZ] = -1.0*c[Z][Z];
	
	ig = normal[1]/2;
	ih = normal[1]%2;
	for (n1 = 0; n1 < NSYMM; n1++) {
	  gradn[n1][ig][ih] = gradn[n1][ig][1 - ih];
	}
	
      }
      
      if (nunknown > 2) {
	
	q_boundary_constants(fe->param, ic, jc, kc, qs, bcs[normal[2]],
			     status[normal[2]], c, cinfo);
	
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
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
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
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[n1] -= bc[n1][n2][normal[2]]*dq;
	    
	  }
	}
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[1]], bc);
	
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
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[n1] *= bcsign[normal[0]];
	}
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[NSYMM + n1] *= bcsign[normal[1]];
	}
	
	gradient_bcs6x6_coeff(kappa0, kappa1, bcs[normal[2]], bc);
	
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
      
      /* Fix the trace (don't care about Qzz in the end) */
      
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
	fg->delsq[addr_rank1(nsites, NQAB, index, n1)] = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  fg->grad[addr_rank2(nsites, NQAB, 3, index, n1, ia)] =
	    0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	  fg->delsq[addr_rank1(nsites, NQAB, index, n1)]
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
 *  gradient_param_commit
 *
 *  Compute and store the inverse matrices used in the boundary conditions.
 *
 *****************************************************************************/

__host__
int gradient_param_commit(grad_lc_anch_t * anch) {

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

    gradient_bcs6x6_coeff(feparam.kappa0, feparam.kappa1, bcs[2*ia + 1], bc);

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

  copyConstToTarget(&static_param, anch->param, sizeof(param_t));

  return 0;
}


/*****************************************************************************
 *
 *  gradient_bcs6x6_coeff
 *
 *  Full set of coefficients in boundary condition equation for given
 *  surface normal dn.
 *
 *****************************************************************************/

__host__ __device__
int gradient_bcs6x6_coeff(double kappa0, double kappa1, const int dn[3],
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
 *  q_boundary_constants
 *
 *  Compute the comstant term in the cholesteric boundary condition.
 *  Fluid point is (ic, jc, kc) with fluid Q_ab = qs
 *  The outward normal is di[3], and the map status is as given.
 *
 *****************************************************************************/

__host__ __device__
int q_boundary_constants(fe_lc_param_t * param, int ic, int jc, int kc,
			 double qs[3][3],
			 const int di[3], int status, double c[3][3],
			 colloids_info_t * cinfo) {
  int index;
  int ia, ib, ig, ih;
  int anchor;
  int noffset[3];

  double w1, w2;
  double dnhat[3];
  double qtilde[3][3];
  double q0[3][3];
  double q2 = 0.0;
  double rd;
  double amp;
  KRONECKER_DELTA_CHAR(d);
  LEVI_CIVITA_CHAR(e);

  colloid_t * pc = NULL;

  /* Default -> outward normal, ie., flat wall */

  w1 = param->w1_wall;
  w2 = param->w2_wall;
  anchor = param->anchoring_wall;

  dnhat[X] = 1.0*di[X];
  dnhat[Y] = 1.0*di[Y];
  dnhat[Z] = 1.0*di[Z];
  fe_lc_amplitude_compute(param, &amp);

  if (status == MAP_COLLOID) {

#ifdef __NVCC__
    assert(0);
#else
    coords_nlocal_offset(noffset);
    index = coords_index(ic - di[X], jc - di[Y], kc - di[Z]);
#endif
    pc = cinfo->map_new[index];
    assert(pc);

    w1 = param->w1_coll;
    w2 = param->w2_coll;
    anchor = param->anchoring_coll;

    dnhat[X] = 1.0*(noffset[X] + ic) - pc->s.r[X];
    dnhat[Y] = 1.0*(noffset[Y] + jc) - pc->s.r[Y];
    dnhat[Z] = 1.0*(noffset[Z] + kc) - pc->s.r[Z];

    /* unit vector */
    rd = 1.0/sqrt(dnhat[X]*dnhat[X] + dnhat[Y]*dnhat[Y] + dnhat[Z]*dnhat[Z]);
    dnhat[X] *= rd;
    dnhat[Y] *= rd;
    dnhat[Z] *= rd;
  }

  if (anchor == LC_ANCHORING_NORMAL) {

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.5*amp*(3.0*dnhat[ia]*dnhat[ib] - d[ia][ib]);
	qtilde[ia][ib] = 0.0;
      }
    }
  }
  else { /* PLANAR */

    q2 = 0.0;
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        qtilde[ia][ib] = qs[ia][ib] + 0.5*amp*d[ia][ib];
        q2 += qtilde[ia][ib]*qtilde[ia][ib];
      }
    }

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.0;
        for (ig = 0; ig < 3; ig++) {
          for (ih = 0; ih < 3; ih++) {
            q0[ia][ib] += (d[ia][ig] - dnhat[ia]*dnhat[ig])*qtilde[ig][ih]
              *(d[ih][ib] - dnhat[ih]*dnhat[ib]);
          }
        }
        q0[ia][ib] -= 0.5*amp*d[ia][ib];
      }
    }
  }

  /* Compute c[a][b] */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      c[ia][ib] = 0.0;

      for (ig = 0; ig < 3; ig++) {
        for (ih = 0; ih < 3; ih++) {
          c[ia][ib] -= param->kappa1*param->q0*di[ig]*
	    (e[ia][ig][ih]*qs[ih][ib] + e[ib][ig][ih]*qs[ih][ia]);
        }
      }

      /* Normal anchoring: w2 must be zero and q0 is preferred Q
       * Planar anchoring: in w1 term q0 is effectively
       *                   (Qtilde^perp - 0.5S_0) while in w2 we
       *                   have Qtilde appearing explicitly.
       *                   See colloids_q_boundary() etc */

      c[ia][ib] +=
	-w1*(qs[ia][ib] - q0[ia][ib])
	-w2*(2.0*q2 - 4.5*amp*amp)*qtilde[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_q_boundary
 *
 *  Produce an estimate of the surface order parameter Q^0_ab for
 *  normal or planar anchoring.
 *
 *  This will depend on the outward surface normal nhat, and in the
 *  case of planar anchoring may depend on the estimate of the
 *  existing order parameter at the surface Qs_ab.
 *
 *  This planar anchoring idea follows e.g., Fournier and Galatola
 *  Europhys. Lett. 72, 403 (2005).
 *
 *****************************************************************************/

__host__ __device__
int colloids_q_boundary(fe_lc_param_t * param,
			const double nhat[3], double qs[3][3],
			double q0[3][3], int map_status) {
  int ia, ib, ic, id;
  int anchoring;

  double qtilde[3][3];
  double amplitude;
  double  nfix[3] = {0.0, 1.0, 0.0};
  KRONECKER_DELTA_CHAR(d);

  assert(map_status == MAP_COLLOID || map_status == MAP_BOUNDARY);

  anchoring = param->anchoring_coll;
  if (map_status == MAP_BOUNDARY) anchoring = param->anchoring_wall;

  fe_lc_amplitude_compute(param, &amplitude);

  if (anchoring == LC_ANCHORING_FIXED) fe_lc_q_uniaxial(param, nfix, q0);
  if (anchoring == LC_ANCHORING_NORMAL) fe_lc_q_uniaxial(param, nhat, q0);

  if (anchoring == LC_ANCHORING_PLANAR) {

    /* Planar: use the fluid Q_ab to find ~Q_ab */

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	qtilde[ia][ib] = qs[ia][ib] + 0.5*amplitude*d[ia][ib];
      }
    }

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	q0[ia][ib] = 0.0;
	for (ic = 0; ic < 3; ic++) {
	  for (id = 0; id < 3; id++) {
	    q0[ia][ib] += (d[ia][ic] - nhat[ia]*nhat[ic])*qtilde[ic][id]
	      *(d[id][ib] - nhat[id]*nhat[ib]);
	  }
	}
	/* Return Q^0_ab = ~Q_ab - (1/2) A d_ab */
	q0[ia][ib] -= 0.5*amplitude*d[ia][ib];
      }
    }

  }

  return 0;
}
