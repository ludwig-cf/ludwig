/****************************************************************************
 *
 *  phi_lb_coupler.c
 *
 *  In cases where the order parameter is via "full LB", this couples
 *  the scalar order parameter field to the 'g' distribution.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "lb_model_s.h"
#include "field_s.h"
#include "phi_lb_coupler.h"

#define NDIST 2

__global__ void phi_lb_to_field_kernel(kernel_ctxt_t * ktxt, field_t * phi,
				       lb_t * lb);

/*****************************************************************************
 *
 *  phi_lb_to_field
 *
 *  Driver function: compute from the distribution the current
 *  values of phi and store.
 *
 *****************************************************************************/

__host__ int phi_lb_to_field(field_t * phi, lb_t  * lb) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(phi);
  assert(lb);

  coords_nlocal(nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  __host_launch(phi_lb_to_field_kernel, nblk, ntpb, ctxt->target,
		phi->target, lb->target);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_lb_to_field_kernel
 *
 *****************************************************************************/

__global__ void phi_lb_to_field_kernel(kernel_ctxt_t * ktx, field_t * phi,
				       lb_t * lb) {
  int kindex;
  __shared__ int kiter;

  assert(ktx);
  assert(phi);
  assert(lb);

  kiter = kernel_iterations(ktx);

  __target_simt_parallel_for(kindex, kiter, 1) {

    int ic, jc, kc, index;
    int p;
    double phi0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = kernel_coords_index(ktx, ic, jc, kc);
    
    phi0 = 0.0;
    for (p = 0; p < NVEL; p++) {
      phi0 += lb->f[LB_ADDR(lb->nsite, NDIST, NVEL, index, LB_PHI, p)];
    }

    phi->data[addr_rank0(phi->nsites, index)] = phi0;
  }

  return;
}

/*****************************************************************************
 *
 *  phi_lb_from_field
 *
 *  Move the scalar order parameter into the non-propagating part
 *  of the distribution, and set other elements of distribution to
 *  zero.
 *
 *****************************************************************************/

__host__ int phi_lb_from_field(field_t * phi, lb_t * lb) {

  int p;
  int ic, jc, kc, index;
  int nlocal[3];

  double phi0;

  assert(phi);
  assert(lb);
  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_scalar(phi, index, &phi0);

	lb_f_set(lb, index, 0, LB_PHI, phi0);
	for (p = 1; p < NVEL; p++) {
	  lb_f_set(lb, index, p, LB_PHI, 0.0);
	}
      }
    }
  }

  return 0;
}
