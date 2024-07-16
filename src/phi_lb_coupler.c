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
 *  (c) 2010-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "phi_lb_coupler.h"

#define NDIST 2

__global__ void phi_lb_to_field_kernel(kernel_3d_t k3d, field_t * phi,
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

  int nlocal[3] = {0};

  assert(phi);
  assert(lb);

  cs_nlocal(phi->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(phi->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(phi_lb_to_field_kernel, nblk, ntpb, 0, 0, k3d,
		    phi->target, lb->target);

    tdpAssert( tdpPeekAtLastError() );
    tdpAssert( tdpDeviceSynchronize() );
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_lb_to_field_kernel
 *
 *****************************************************************************/

__global__ void phi_lb_to_field_kernel(kernel_3d_t k3d, field_t * phi,
				       lb_t * lb) {
  int kindex = 0;

  assert(phi);
  assert(lb);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    double phi0 = 0.0;
    for (int p = 0; p < NVEL; p++) {
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

  cs_nlocal(phi->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(phi->cs, ic, jc, kc);

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
