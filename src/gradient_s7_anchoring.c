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
#include "util_vector.h"
#include "util_ellipsoid.h"
#include "coords.h"
#include "kernel.h"
#include "colloids.h"

#include "lc_anchoring.h"
#include "gradient_s7_anchoring.h"

struct grad_s7_anch_s {
  pe_t * pe;
  cs_t * cs;
  lc_anchoring_matrices_t bc;/* Boundary condition parameters */
  map_t * map;               /* Supports a map */
  fe_lc_t * fe;              /* Liquid crystal free energy */
  colloids_info_t * cinfo;   /* Colloid information */
  grad_s7_anch_t * target;   /* Device memory */
};

static grad_s7_anch_t * static_grad = NULL;


__global__
void grad_s7_kernel(kernel_ctxt_t * ktx, cs_t * cs, grad_s7_anch_t * anch,
		    fe_lc_t * fe, field_grad_t * fg, map_t * map);

__host__ __device__ int grad_s7_boundary_c(fe_lc_param_t * param,
					   grad_s7_anch_t * anch,
					   int ic, int jc, int kc,
					   int status,
					   const double qs[3][3],
					   const int di[3],
					   double c[3][3]);

/*****************************************************************************
 *
 *  grad_s7_anchoring_create
 *
 *  Note we just attach the new object to the static pointer at the
 *  moment.
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

  obj->pe = pe;
  obj->cs = cs;
  obj->map = map;
  obj->fe = fe;

  {
    fe_lc_param_t fep = {0};
    fe_lc_param(fe, &fep);
    lc_anchoring_matrices(fep.kappa0, fep.kappa1, &obj->bc);
  }

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    cs_t * cstarget = NULL;
    tdpMalloc((void **) &obj->target, sizeof(grad_s7_anch_t));
    tdpMemset(obj->target, 0, sizeof(grad_s7_anch_t));

    cs_target(obj->cs, &cstarget);
    tdpMemcpy(&obj->target->cs, &cstarget, sizeof(cs_t *),
              tdpMemcpyHostToDevice);

    tdpAssert(tdpMemcpy(&obj->target->bc, &obj->bc,
			sizeof(lc_anchoring_matrices_t),
			tdpMemcpyHostToDevice));
  }

  static_grad = obj;

  return 0;
}

/*****************************************************************************
 *
 *  grad_s7_anchoring_cinfo_set
 *
 *****************************************************************************/

__host__ int grad_s7_anchoring_cinfo_set(colloids_info_t * cinfo) {

  int ndevice = 0;

  assert(cinfo);

  static_grad->cinfo = cinfo;

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    tdpAssert(tdpMemcpy(&static_grad->target->cinfo, &cinfo->target,
			sizeof(colloids_info_t *), tdpMemcpyHostToDevice));
  }

  /* Any ellispsoidal particles must have b == c */

  {
    colloid_t * pc = NULL;

    colloids_info_local_head(cinfo, &pc);

    for ( ; pc; pc = pc->nextlocal) {
      if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
	if (fabs(pc->s.elabc[1] - pc->s.elabc[2]) > DBL_EPSILON) {
	  pe_info(cinfo->pe, "s7_anchoring: ellispoid must have b == c\n");
	  pe_fatal(cinfo->pe, "Please check te input and try again\n");
	}
      }
    }
  }

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

  free(grad);

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

  fe_lc_param_commit(anch->fe);

  cs_target(anch->cs, &cstarget);

  tdpLaunchKernel(grad_s7_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, cstarget,
		  anch->target, anch->fe->target, fg->target,
		  anch->map->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

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
	
	grad_s7_boundary_c(fe->param, anch, ic, jc, kc, status[normal[0]], qs,
			   bcs[normal[0]], c);
	
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
	/* Cases to trap: status[0] != status[1] (e.g., one wall, one
         * colloid). Default to MAP_BOUNDARY, then check have same
	 * status. */

	int bcse[3] = {0, 0, 0};
	int nn0 = normal[0];
	int nn1 = normal[1];
	int mystatus = MAP_BOUNDARY;

	bcse[X] = bcs[nn0][X] + bcs[nn1][X];
	bcse[Y] = bcs[nn0][Y] + bcs[nn1][Y];
	bcse[Z] = bcs[nn0][Z] + bcs[nn1][Z];

	if (status[nn0] == status[nn1]) mystatus = status[nn0];

	grad_s7_boundary_c(fe->param, anch, ic, jc, kc, mystatus, qs, bcse, c);

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
	int mystatus = MAP_BOUNDARY;

	bcse[X] = bcs[nn0][X] + bcs[nn1][X] + bcs[nn2][X];
	bcse[Y] = bcs[nn0][Y] + bcs[nn1][Y] + bcs[nn2][Y];
	bcse[Z] = bcs[nn0][Z] + bcs[nn1][Z] + bcs[nn2][Z];

	if (status[nn0] == status[nn1] && status[nn0] == status[nn2]) {
	  mystatus = status[nn0];
	}

	grad_s7_boundary_c(fe->param, anch, ic, jc, kc, mystatus, qs, bcse, c);

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
 *  grad_s7_boundary_coll
 *
 *  Compute the constant term in the cholesteric boundary condition.
 *  Fluid Q_ab = qs
 *
 *  The outward normal is dr[3].
 *
 *****************************************************************************/

__host__ __device__ void grad_s7_boundary_coll(fe_lc_param_t * param,
					       const double qs[3][3],
					       const double r[3],
					       double c[3][3]) {

  double rhat[3] = {r[X], r[Y], r[Z]};
  double amp     = 0.0;
  double kappa1  = param->kappa1;
  double q0      = param->q0;

  fe_lc_amplitude_compute(param, &amp);

  /* Make sure we have a unit vector */
  {
    double rd = 1.0/sqrt(rhat[X]*rhat[X] + rhat[Y]*rhat[Y] + rhat[Z]*rhat[Z]);
    rhat[X] *= rd;
    rhat[Y] *= rd;
    rhat[Z] *= rd;
  }

  {
    lc_anchoring_param_t * anchor = &param->coll;

    if (anchor->type == LC_ANCHORING_NORMAL) {
      lc_anchoring_normal_ct(anchor, qs, rhat, kappa1, q0, amp, c);
    }

    if (anchor->type == LC_ANCHORING_PLANAR) {
      lc_anchoring_planar_ct(anchor, qs, rhat, kappa1, q0, amp, c);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  grad_s7_boundary_wall
 *
 *  Compute the constant term in the cholesteric boundary condition.
 *  Fluid Q_ab = qs.
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
 *  grad_s7_anchoring_bc
 *
 *  This just switches depending on status boundary or colloid.
 *
 *****************************************************************************/

__host__ __device__ int grad_s7_boundary_c(fe_lc_param_t * param,
					   grad_s7_anch_t * anch,
					   int ic, int jc, int kc,
					   int status,
					   const double qs[3][3],
					   const int di[3],
					   double c[3][3]) {

  assert(status != MAP_FLUID);

  if (status == MAP_BOUNDARY) {
    grad_s7_boundary_wall(param, qs, di, c);
  }
  else if (status == MAP_COLLOID) {
    /* Compute relevant vector from colloid centre to fluid site. */
    /* Fluid site (ic,jc,kc); colloid site is fluid site - di[] */
    /* Use fluid site as the boundary position. */

    int noffset[3] = {0};
    double dr[3] = {0};
    cs_nlocal_offset(anch->cs, noffset);
    {
      int index = cs_index(anch->cs, ic - di[X], jc - di[Y], kc - di[Z]);
      colloid_t * pc = anch->cinfo->map_new[index];

      if (pc) {
	dr[X] = 1.0*(noffset[X] + ic) - pc->s.r[X];
	dr[Y] = 1.0*(noffset[Y] + jc) - pc->s.r[Y];
	dr[Z] = 1.0*(noffset[Z] + kc) - pc->s.r[Z];

	if (pc->s.shape == COLLOID_SHAPE_ELLIPSOID) {
	  int isphere = util_ellipsoid_is_sphere(pc->s.elabc);
	  if (!isphere) {
	    double rs[3] = {0};
	    util_vector_copy(3, dr, rs);
	    util_spheroid_surface_normal(pc->s.elabc, pc->s.m, rs, dr);
	  }
	} 
	grad_s7_boundary_coll(param, qs, dr, c);
      }
      else {
	/* There is a case to trap here, where a diagonal di[] crosses
	 * the gap between two different colloids. This will give a
	 * fluid site. It's not clear there is any 'correct' answer
	 * to what to do in this case as there is no unique surface normal.
	 * So set c = 0. */
	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    c[ia][ib] = 0.0;
	  }
	}
      }
    }
  }
  else {
    /* Should not be here. */
    assert(0);
  }

  return 0;
}
