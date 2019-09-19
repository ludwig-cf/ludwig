/*****************************************************************************
 *
 *  gradient_2d_ternary_solid.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
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
#include "gradient_3d_27pt_solid.h"

struct solid_s {
  map_t * map;
  fe_ternary_t * fe;
};

static struct solid_s static_solid = {0};

/* These are the 'links' used to form the gradients at boundaries. */

#define NGRAD_ 9
static __constant__ int bs_cv[NGRAD_][2] = {{ 0, 0},
				 {-1,-1}, {-1, 0}, {-1, 1},
                                 { 0,-1}, { 0, 1}, { 1,-1},
				 { 1, 0}, { 1, 1}};


/*****************************************************************************
 *
 *  grad_2d_ternary_solid_map_set
 *
 *****************************************************************************/

__host__ int grad_2d_ternary_solid_map_set(map_t * map) {

  int ndata;
  assert(map);

  static_solid.map = map;

  /* We expect at most two wetting parameters; if present
   * first should be C, second H. Default to zero. */

  map_ndata(map, &ndata);
  if (ndata < 2) pe_fatal(map->pe, "Wetting parameters%d\n", ndata);

  return 0;
}

/*****************************************************************************
 *
 *  grad_2d_ternary_solid_fe_set
 *
 *****************************************************************************/

__host__ int grad_ternary_solid_fe_set(fe_ternary_t * fe) {

  assert(fe);

  static_solid.fe = fe;

  return 0;
}

/*****************************************************************************
 *
 *  grad_ternary_solid_d2
 *
 *****************************************************************************/

__host__ int grad_ternary_solid_d2(field_grad_t * fgrad) {

  int nextra;
  int nlocal[3];
  double rkappa;
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  fe_symm_param_t param;

  cs_nhalo(fgrad->field->cs, &nextra);
  nextra -= 1;
  cs_nlocal(fgrad->field->cs, nlocal);

  assert(nextra >= 0);
  assert(static_solid.map);
  assert(static_solid.fe_symm);

  fe_symm_param(static_solid.fe_symm, &param);
  rkappa = 1.0/param.kappa;

  limits.imin = 1 - nextra; limits.imax = nlocal[X] + nextra;
  limits.jmin = 1 - nextra; limits.jmax = nlocal[Y] + nextra;
  limits.kmin = 1 - nextra; limits.kmax = nlocal[Z] + nextra;

  kernel_ctxt_create(fgrad->field->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(grad_2d_ternary_solid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target,
		  fgrad->target, static_solid.map->target, rkappa);
  tdpDeviceSynchronize();

  kernel_ctxt_free(ctxt);

  return 0;
}

/****************************************************************************
 *
 *  grad_2d_ternary_solid_kernel
 *
 *  kappa is the interfacial energy penalty in the symmetric picture.
 *  rkappa is (1.0/kappa). 
 *
 ****************************************************************************/

__global__ void grad_3d_27pt_solid_kernel(kernel_ctxt_t * ktx,
					  field_grad_t * fg,
					  map_t * map,
					  double * kappa) {
  int kindex;
  int kiterations;
  const double r9 = (1.0/9.0);     /* normaliser for grad */
  const double r18 = (1.0/18.0);   /* normaliser for delsq */


  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int nf;
    int ic, jc, kc, ic1, jc1, kc1;
    int ia, index, p;
    int n;

    int isite[NGRAD_];

    double count[NGRAD_];
    double gradt[NGRAD_];
    double gradn[3];
    double dphi;
    double c, h, phi_b;

    int status;
    double wet[2] = {0.0, 0.0};

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
	if (status != MAP_FLUID) isite[p] = MAP_SOLID;
      }

      for (n = 0; n < nf; n++) {

	for (ia = 0; ia < 3; ia++) {
	  count[ia] = 0.0;
	  gradn[ia] = 0.0;
	}
	  
	for (p = 1; p < NGRAD_; p++) {

	  if (isite[p] != MAP_FLUID) continue;

	  dphi = phi->data[addr_rank1(phi->nsites, nop, isite[p], n)]
	       - phi->data[addr_rank1(phi->nsites, nop, index,    n)];

	  gradn[X] += 3.0*wv[p]*bs_cv[p][X]*dphi;
	  gradn[Y] += 3.0*wv[p]*bs_cv[p][Y]*dphi;

	  for (ia = 0; ia < 2; ia++) {
	    count[ia] += bs_cv[p][ia]*bs_cv[p][ia];
	  }
	}

	/* Estimate gradient at boundaries */
 
	/* Accumulate the final gradients */

	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,X)] = gradn[X];
	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,Y)] = gradn[Y];
	fg->grad[addr_rank2(phi->nsites,nf,3,index,n,Z)] = 0.0;
      }

      /* Next fluid site */
    }
  }

  return;
}

/*****************************************************************************
 *
 *  grad_3d_27pt_solid_dab
 *
 *****************************************************************************/

__host__ int grad_3d_27pt_solid_dab(field_grad_t * df) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int nsites;
  int ic, jc, kc;
  int xs, ys, zs;
  int index;
  double * __restrict__ dab;
  double * __restrict__ field;

  cs_t * cs = NULL;

  const double r12 = (1.0/12.0);

  assert(df);
  assert(cs);

  cs_nhalo(cs, &nhalo);
  cs_nlocal(cs, nlocal);
  cs_nsites(cs, &nsites);
  cs_strides(cs, &xs, &ys, &zs);

  nextra = nhalo - 1;
  assert(nextra >= 0);

  field = df->field->data;
  dab = df->d_ab;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

        index = cs_index(cs, ic, jc, kc);

        dab[addr_rank1(nsites, NSYMM, index, XX)] = 0.2*
         (+ 1.0*field[addr_rank0(nsites, index + xs)]
          + 1.0*field[addr_rank0(nsites, index - xs)]
          - 2.0*field[addr_rank0(nsites, index)]
          + 1.0*field[addr_rank0(nsites, index + xs + ys)]
          + 1.0*field[addr_rank0(nsites, index - xs + ys)]
          - 2.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index + xs - ys)]
          + 1.0*field[addr_rank0(nsites, index - xs - ys)]
          - 2.0*field[addr_rank0(nsites, index - ys)]
          + 1.0*field[addr_rank0(nsites, index + xs + 1)]
          + 1.0*field[addr_rank0(nsites, index - xs + 1)]
          - 2.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index + xs - 1)]
          + 1.0*field[addr_rank0(nsites, index - xs - 1)]
          - 2.0*field[addr_rank0(nsites, index - 1)]);

        dab[addr_rank1(nsites, NSYMM, index, XY)] = r12*
          (+ field[addr_rank0(nsites, index + xs + ys)]
           - field[addr_rank0(nsites, index + xs - ys)]
           - field[addr_rank0(nsites, index - xs + ys)]
           + field[addr_rank0(nsites, index - xs - ys)]
           + field[addr_rank0(nsites, index + xs + ys + 1)]
           - field[addr_rank0(nsites, index + xs - ys + 1)]
           - field[addr_rank0(nsites, index - xs + ys + 1)]
           + field[addr_rank0(nsites, index - xs - ys + 1)]
           + field[addr_rank0(nsites, index + xs + ys - 1)]
           - field[addr_rank0(nsites, index + xs - ys - 1)]
           - field[addr_rank0(nsites, index - xs + ys - 1)]
           + field[addr_rank0(nsites, index - xs - ys - 1)]);

        dab[addr_rank1(nsites, NSYMM, index, XZ)] = r12*
          (+ field[addr_rank0(nsites, index + xs + 1)]
           - field[addr_rank0(nsites, index + xs - 1)]
           - field[addr_rank0(nsites, index - xs + 1)]
           + field[addr_rank0(nsites, index - xs - 1)]
           + field[addr_rank0(nsites, index + xs + ys + 1)]
           - field[addr_rank0(nsites, index + xs + ys - 1)]
           - field[addr_rank0(nsites, index - xs + ys + 1)]
           + field[addr_rank0(nsites, index - xs + ys - 1)]
           + field[addr_rank0(nsites, index + xs - ys + 1)]
           - field[addr_rank0(nsites, index + xs - ys - 1)]
           - field[addr_rank0(nsites, index - xs - ys + 1)]
           + field[addr_rank0(nsites, index - xs - ys - 1)]);

        dab[addr_rank1(nsites, NSYMM, index, YY)] = 0.2*
         (+ 1.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys)]
          - 2.0*field[addr_rank0(nsites, index)]
          + 1.0*field[addr_rank0(nsites, index + xs + ys)]
          + 1.0*field[addr_rank0(nsites, index + xs - ys)]
          - 2.0*field[addr_rank0(nsites, index + xs)]
          + 1.0*field[addr_rank0(nsites, index - xs + ys)]
          + 1.0*field[addr_rank0(nsites, index - xs - ys)]
          - 2.0*field[addr_rank0(nsites, index - xs)]
          + 1.0*field[addr_rank0(nsites, index + 1 + ys)]
          + 1.0*field[addr_rank0(nsites, index + 1 - ys)]
          - 2.0*field[addr_rank0(nsites, index + 1 )]
          + 1.0*field[addr_rank0(nsites, index - 1 + ys)]
          + 1.0*field[addr_rank0(nsites, index - 1 - ys)]
          - 2.0*field[addr_rank0(nsites, index - 1 )]);


        dab[addr_rank1(nsites, NSYMM, index, YZ)] = r12*
          (+ field[addr_rank0(nsites, index + ys + 1)]
           - field[addr_rank0(nsites, index + ys - 1)]
           - field[addr_rank0(nsites, index - ys + 1)]
           + field[addr_rank0(nsites, index - ys - 1)]
           + field[addr_rank0(nsites, index + xs + ys + 1)]
           - field[addr_rank0(nsites, index + xs + ys - 1)]
           - field[addr_rank0(nsites, index + xs - ys + 1)]
           + field[addr_rank0(nsites, index + xs - ys - 1)]
           + field[addr_rank0(nsites, index - xs + ys + 1)]
           - field[addr_rank0(nsites, index - xs + ys - 1)]
           - field[addr_rank0(nsites, index - xs - ys + 1)]
           + field[addr_rank0(nsites, index - xs - ys - 1)]);

        dab[addr_rank1(nsites, NSYMM, index, ZZ)] = 0.2*
         (+ 1.0*field[addr_rank0(nsites, index + 1)]
          + 1.0*field[addr_rank0(nsites, index - 1)]
          - 2.0*field[addr_rank0(nsites, index)]
          + 1.0*field[addr_rank0(nsites, index + xs + 1)]
          + 1.0*field[addr_rank0(nsites, index + xs - 1)]
          - 2.0*field[addr_rank0(nsites, index + xs)]
          + 1.0*field[addr_rank0(nsites, index - xs + 1)]
          + 1.0*field[addr_rank0(nsites, index - xs - 1)]
          - 2.0*field[addr_rank0(nsites, index - xs)]
          + 1.0*field[addr_rank0(nsites, index + ys + 1)]
          + 1.0*field[addr_rank0(nsites, index + ys - 1)]
          - 2.0*field[addr_rank0(nsites, index + ys)]
          + 1.0*field[addr_rank0(nsites, index - ys + 1)]
          + 1.0*field[addr_rank0(nsites, index - ys - 1)]
          - 2.0*field[addr_rank0(nsites, index - ys)]);

      }
    }
  }

  return 0;
}
