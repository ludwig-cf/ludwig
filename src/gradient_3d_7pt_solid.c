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
 *  Experimental feature.
 *  Depedence on the compositional order parameter phi is introduced
 *  to allow wetting in the LC droplet case.
 * 
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2023 The University of Edinburgh
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
#include "coords.h"
#include "kernel.h"
#include "colloids.h"

#include "util.h"
#include "lc_anchoring.h"
#include "gradient_3d_7pt_solid.h"

struct grad_lc_anch_s {
  pe_t * pe;
  cs_t * cs;
  lc_anchoring_matrices_t bc;/* Boundary condition matrices */
  map_t * map;               /* Supports a map */
  field_t * phi;             /* Compositional order parameter */
  colloids_info_t * cinfo;   /* Supports colloids */
  fe_lc_t * fe;              /* Liquid crystal free energy */
  grad_lc_anch_t * target;   /* Device memory */
};

static grad_lc_anch_t * static_grad = NULL;

__host__ int gradient_6x6(grad_lc_anch_t * anch, field_grad_t *grad,
			  int nextra);

__global__
void gradient_6x6_kernel(kernel_ctxt_t * ktx, cs_t * cs, grad_lc_anch_t * anch,
			 fe_lc_t * fe, field_grad_t * fg,
			 map_t * map, colloids_info_t * cinfo);

__host__ __device__ int grad_3d_7pt_bc(grad_lc_anch_t * anch,
				       fe_lc_param_t * fep,
				       int ic, int jc, int kc,
				       int status,
				       const double qs[3][3],
				       const int di[3],
				       double c[3][3]);

/*****************************************************************************
 *
 *  grad_lc_anch_create
 *
 *  phi may be NULL, in which case this is the bare LC case.
 *
 *****************************************************************************/

__host__ int grad_lc_anch_create(pe_t * pe, cs_t * cs, map_t * map,
				 field_t * phi, colloids_info_t * cinfo,
				 fe_lc_t * fe, grad_lc_anch_t ** pobj) {
  int ndevice;
  grad_lc_anch_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(fe);

  obj = (grad_lc_anch_t *) calloc(1, sizeof(grad_lc_anch_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(grad_lc_anch_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->map = map;
  obj->phi = phi;
  obj->cinfo = cinfo;
  obj->fe = fe;

  {
    /* Initialise matrix inverses */
    fe_lc_param_t fep = {0};
    fe_lc_param(fe, &fep);
    lc_anchoring_matrices(fep.kappa0, fep.kappa1, &obj->bc);
  }

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    /* Copy required entities over ... */
    cs_t * tcs = NULL;
    tdpMalloc((void **) &obj->target, sizeof(grad_lc_anch_t));

    cs_target(obj->cs, &tcs);
    tdpAssert(tdpMemcpy(&obj->target->cs, &tcs, sizeof(cs_t *),
			tdpMemcpyHostToDevice));

    if (cinfo) {
    tdpAssert(tdpMemcpy(&obj->target->cinfo, &cinfo->target,
			sizeof(colloids_info_t *), tdpMemcpyHostToDevice));
    }

    tdpAssert(tdpMemcpy(&obj->target->bc, &obj->bc,
			sizeof(lc_anchoring_matrices_t),
			tdpMemcpyHostToDevice));
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

  if (grad->target != grad) tdpFree(grad->target);
  free(grad);

  return 0;
}

/*****************************************************************************
 *
 *  grad_3d_7pt_solid_set
 *
 *****************************************************************************/

__host__ int grad_3d_7pt_solid_set(map_t * map, colloids_info_t * cinfo) {

  int ndevice = 0;

  assert(map);
  assert(static_grad);

  static_grad->map = map;
  static_grad->cinfo = cinfo;

  tdpGetDeviceCount(&ndevice);
  if (ndevice) {
    tdpAssert(tdpMemcpy(&static_grad->target->cinfo, &cinfo->target,
			sizeof(colloids_info_t *), tdpMemcpyHostToDevice));
  }

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

  int nhalo;
  int nextra;
  grad_lc_anch_t * grad = NULL; /* get static instance and use */

  assert(static_grad);
  assert(static_grad->fe);

  grad = static_grad;

  cs_nhalo(grad->cs, &nhalo);
  nextra = nhalo - 1;

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
  cs_t * cstarget = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(anch);
  assert(fg);

  cs_nlocal(anch->cs, nlocal);

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(anch->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  fe_lc_param_commit(anch->fe);
  cs_target(anch->cs, &cstarget);

  tdpLaunchKernel(gradient_6x6_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, cstarget,
		  anch->target, anch->fe->target, fg->target,
		  anch->map->target, anch->cinfo->target);
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
void gradient_6x6_kernel(kernel_ctxt_t * ktx, cs_t * cs, grad_lc_anch_t * anch,
			 fe_lc_t * fe, field_grad_t * fg,
			 map_t * map, colloids_info_t * cinfo) {

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
	qs[Z][Z] = 0.0
	  - q->data[addr_rank1(q->nsites, NQAB, index, XX)]
	  - q->data[addr_rank1(q->nsites, NQAB, index, YY)];

	grad_3d_7pt_bc(anch, fe->param, ic, jc, kc, status[normal[0]], qs,
		       bcs[normal[0]], c);

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

	grad_3d_7pt_bc(anch, fe->param, ic, jc, kc, status[normal[1]], qs,
		       bcs[normal[1]], c);

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

	grad_3d_7pt_bc(anch, fe->param, ic, jc, kc, status[normal[2]], qs,
		       bcs[normal[2]], c);

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
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[0]], bc);
	
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
	  x18[n1] = anch->bc.a6inv[normal[0]/2][n1]*b18[n1];
	}
      }
      
      if (nunknown == 2) {
	
	if (normal[0]/2 == X && normal[1]/2 == Y) normal[2] = Z;
	if (normal[0]/2 == X && normal[1]/2 == Z) normal[2] = Y;
	if (normal[0]/2 == Y && normal[1]/2 == Z) normal[2] = X;
	
	/* Compute the RHS for two unknowns and one known */
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
	    b18[n1] -= bc[n1][n2][normal[2]]*dq;
	    
	  }
	}
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[1]], bc);
	
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
	    x18[n1] += bcsign[normal[0]]*anch->bc.a12inv[ia][n1][n2]*b18[n2];
	  }
	  for (n2 = NSYMM; n2 < 2*NSYMM; n2++) {
	    x18[n1] += bcsign[normal[1]]*anch->bc.a12inv[ia][n1][n2]*b18[n2];
	  }
	}
      }
      
      if (nunknown == 3) {
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[0]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[n1] *= bcsign[normal[0]];
	}
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[1]], bc);
	
	for (n1 = 0; n1 < NSYMM; n1++) {
	  for (n2 = 0; n2 < NSYMM; n2++) {
	    dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;
	    
	    dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	    b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
	  }
	  b18[NSYMM + n1] *= bcsign[normal[1]];
	}
	
	lc_anchoring_coefficients(kappa0, kappa1, bcs[normal[2]], bc);
	
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
	    x18[n1] += anch->bc.a18inv[n1][n2]*b18[n2];
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
 *  grad_3d_7pt_bc
 *
 *  Driver to compute the constant term c_ab in the boundary consition
 *  dependent on the relevant surface free energy.
 *
 *  (1) Watch out for the different unit vectors here. di[] is the
 *      lattice vector at the outward face; dnhat is relative to
 *      colloid centre in the colloid case and is used to compute the
 *      preferred surface Q_0. This was the choice at
 *      the time this was first implemented.
 *
 *  (2) The "s7" method, in contrast, uses dnhat for both purposes,
 *      and so may be a more consistent choice.
 *
 *****************************************************************************/

__host__ __device__ int grad_3d_7pt_bc(grad_lc_anch_t * anch,
				       fe_lc_param_t * fep,
				       int ic, int jc, int kc,
				       int status,
				       const double qs[3][3],
				       const int di[3],
				       double c[3][3]) {
  assert(anch);
  assert(fep);

  int anchor;
  int noffset[3];

  double w1, w2;
  double dnhat[3];
  double qtilde[3][3] = {0};
  double q0[3][3];
  double q2 = 0.0;
  double rd;
  double amp;
  KRONECKER_DELTA_CHAR(d);

  fe_lc_amplitude_compute(fep, &amp);

  /* Default -> outward normal, ie., flat wall */

  w1 = fep->wall.w1;
  w2 = fep->wall.w2;
  anchor = fep->wall.type;

  dnhat[X] = 1.0*di[X];
  dnhat[Y] = 1.0*di[Y];
  dnhat[Z] = 1.0*di[Z];

  if (status == MAP_COLLOID) {

    int index = cs_index(anch->cs, ic - di[X], jc - di[Y], kc - di[Z]);
    colloid_t * pc = anch->cinfo->map_new[index];

    cs_nlocal_offset(anch->cs, noffset);

    assert(pc);

    w1 = fep->coll.w1;
    w2 = fep->coll.w2;
    anchor = fep->coll.type;

    dnhat[X] = 1.0*(noffset[X] + ic) - pc->s.r[X];
    dnhat[Y] = 1.0*(noffset[Y] + jc) - pc->s.r[Y];
    dnhat[Z] = 1.0*(noffset[Z] + kc) - pc->s.r[Z];

    /* unit vector */
    rd = 1.0/sqrt(dnhat[X]*dnhat[X] + dnhat[Y]*dnhat[Y] + dnhat[Z]*dnhat[Z]);
    dnhat[X] *= rd;
    dnhat[Y] *= rd;
    dnhat[Z] *= rd;
  }



  if (anchor == LC_ANCHORING_FIXED) {
    /* Wall anchoring only. */
    lc_anchoring_fixed_ct(&fep->wall, qs, dnhat, fep->kappa1, fep->q0, amp, c);
  }

  if (anchor == LC_ANCHORING_NORMAL) {
    /* Here's the latice unit vector (note dnhat used for preferred Q_0) */
    double nhat[3] = {1.0*di[X], 1.0*di[Y], 1.0*di[Z]};

    lc_anchoring_normal_q0(dnhat, amp, q0);
    lc_anchoring_kappa1_ct(fep->kappa1, fep->q0, nhat, qs, c);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	c[ia][ib] += -w1*(qs[ia][ib] - q0[ia][ib]);
      }
    }
  }

  if (anchor == LC_ANCHORING_PLANAR) {
    /* See note above */
    double hat[3] = {1.0*di[X], 1.0*di[Y], 1.0*di[Z]};

    q2 = 0.0;
    lc_anchoring_planar_qtilde(amp, qs, qtilde);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.0;
	q2 += qtilde[ia][ib]*qtilde[ia][ib];
        for (int ig = 0; ig < 3; ig++) {
          for (int ih = 0; ih < 3; ih++) {
            q0[ia][ib] += (d[ia][ig] - dnhat[ia]*dnhat[ig])*qtilde[ig][ih]
              *(d[ih][ib] - dnhat[ih]*dnhat[ib]);
          }
        }
        q0[ia][ib] -= 0.5*amp*d[ia][ib];
      }
    }

    /* Compute c[a][b] */

    lc_anchoring_kappa1_ct(fep->kappa1, fep->q0, hat, qs, c);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	c[ia][ib] +=
	  -w1*(qs[ia][ib] - q0[ia][ib])
	  -w2*(2.0*q2 - 4.5*amp*amp)*qtilde[ia][ib];
      }
    }
  }

  /* Experimental liquid crystal emulsion term */

  if (anch->phi) {

    /* Compositional order parameter for LC wetting:
     * The LC anchoring strengths w1 and w2 vanish in the disordered phase.
     * We assume this is the phase which has a negative binary
     * order parameter e.g. phi = -1.
     * The standard anchoring case corresponds to phi = +1 */

    int index   = cs_index(anch->cs, ic, jc, kc);
    double phi  = anch->phi->data[addr_rank0(anch->phi->nsites, index)];
    double wphi = 0.5*(1.0 + phi);

    /* Just an additional factor of wphi ... */
    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	c[ia][ib] *= wphi;
      }
    }
  }

  return 0;
}
