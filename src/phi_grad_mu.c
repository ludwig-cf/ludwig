/*****************************************************************************
 *
 *  phi_grad_mu.c
 *
 *  Various implementations of the computation of the local body force
 *  on the fluid via f_a = -phi \nalba_a mu.
 *
 *  Relevant for Cahn-Hilliard (symmetric, Brazovskii, ... free energies).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Jurij Sablic (jurij.sablic@gmail.com) introduced the external chemical
 *  potential gradient.
 *
 *****************************************************************************/

#include <assert.h>

#include "physics.h"
#include "phi_grad_mu.h"

__global__ void phi_grad_mu_fluid_kernel(kernel_ctxt_t * ktx, field_t * phi,
					 fe_t * fe, hydro_t * hydro);
__global__ void phi_grad_mu_solid_kernel(kernel_ctxt_t * ktx, field_t * phi,
					 fe_t * fe, hydro_t * hydro,
					 map_t * map);
__global__ void phi_grad_mu_external_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    double3 grad_mu, hydro_t * hydro);

__global__ void phi_grad_mu_correction_kernel(kernel_ctxt_t * ktx,
					      field_t * phi, fe_t * fe,
					      hydro_t * hydro, map_t * map,
					      double * fcorrect);

__global__ void phi_grad_mu_force_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
					 map_t * map, double3 fcorrect);

/*****************************************************************************
 *
 *  phi_grad_mu_fluid
 *
 *  Driver for fluid only for given chemical potential (from abstract
 *  free energy description).
 *
 *****************************************************************************/

__host__ int phi_grad_mu_fluid(cs_t * cs, field_t * phi, fe_t * fe,
			       hydro_t * hydro) {
  int nlocal[3];
  dim3 nblk, ntpb;

  fe_t * fe_target;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);

  cs_nlocal(cs, nlocal);
  fe->func->target(fe, &fe_target);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_grad_mu_fluid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, fe_target, hydro->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_grad_mu_solid
 *
 *  Driver for phi grad mu in the presence of solid described by map_t.
 *
 *****************************************************************************/

__host__ int phi_grad_mu_solid(cs_t * cs, field_t * phi, fe_t * fe,
			       hydro_t * hydro, map_t * map) {
  int nlocal[3];
  dim3 nblk, ntpb;

  fe_t * fe_target;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);
  assert(map);

  cs_nlocal(cs, nlocal);
  fe->func->target(fe, &fe_target);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_grad_mu_solid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, fe_target, hydro->target,
		  map->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_grad_mu_external
 *
 *  Driver to accumulate the force originating from external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__host__ int phi_grad_mu_external(cs_t * cs, field_t * phi, hydro_t * hydro) {

  int nlocal[3];
  int is_grad_mu = 0;     /* Short circuit the kernel if not required. */
  dim3 nblk, ntpb;
  double3 grad_mu = {0.0,0.0,0.0};

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);

  cs_nlocal(cs, nlocal);

  {
    physics_t * phys = NULL;
    double mu[3] = {0};

    physics_ref(&phys);
    physics_grad_mu(phys, mu);
    grad_mu.x = mu[X];
    grad_mu.y = mu[Y];
    grad_mu.z = mu[Z];
    is_grad_mu = (mu[X] != 0.0 || mu[Y] != 0.0 || mu[Z] != 0.0);
  }

  /* We may need to revisit the external chemical potential if required
   * for more than one order parameter. */

  if (is_grad_mu && phi->nf == 1) {

    limits.imin = 1; limits.imax = nlocal[X];
    limits.jmin = 1; limits.jmax = nlocal[Y];
    limits.kmin = 1; limits.kmax = nlocal[Z];

    kernel_ctxt_create(cs, 1, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(phi_grad_mu_external_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, phi->target, grad_mu, hydro->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_grad_mu_correction
 *
 *  Driver for a general case which allows for arbitrary solid, and
 *  also provides an option for a correction to any deviation from
 *  \sum f_i /= 0 in the global picture.
 *
 *  The correction is a two-stage process in which any non-zero f_i
 *  is evenly distributed between all available fluid nodes.
 *
 *****************************************************************************/

__host__ int phi_grad_mu_correction(cs_t * cs, field_t * phi, fe_t * fe,
				    hydro_t * hydro, map_t * map, int opt) {

  int nlocal[3] = {0};
  dim3 nblk, ntpb;

  size_t sz = 4*sizeof(double);
  double * flocal_d = NULL;        /* Local contribution to correction */

  fe_t * fe_target = NULL;
  kernel_info_t limits = {0};
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);
  assert(map);

  cs_nlocal(cs, nlocal);
  fe->func->target(fe, &fe_target);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  /* Allocate and initialise the accumulator */
  tdpAssert(tdpMalloc((void **) &flocal_d, sz));
  tdpAssert(tdpMemset(flocal_d, 0, sz));

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_grad_mu_correction_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, fe_target, hydro->target,
		  map->target, flocal_d);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  /* Accumulate the total (all ranks) */

  if (opt == 0) {
    /* No correction */
  }
  else {

    /* Compute and apply the global correction */
    /* The volume ftotal[0] should be an integer number of fluid sites,
     * but is accumulated as a double ... */

    double   ftotal[4] = {0};        /* (volume, fx, fy, fz) */
    MPI_Comm comm = MPI_COMM_NULL;

    cs_cart_comm(cs, &comm);
    tdpAssert(tdpMemcpy(ftotal, flocal_d, sz, tdpMemcpyDeviceToHost));
    MPI_Allreduce(MPI_IN_PLACE, ftotal, 4, MPI_DOUBLE, MPI_SUM, comm);

    if (ftotal[0] > 0.0) {
      ftotal[1] = -ftotal[1]/ftotal[0];
      ftotal[2] = -ftotal[2]/ftotal[0];
      ftotal[3] = -ftotal[3]/ftotal[0];
    }

    {
      double3 f = {ftotal[1], ftotal[2], ftotal[3]};
      tdpLaunchKernel(phi_grad_mu_force_kernel, nblk, ntpb, 0, 0,
		      ctxt->target, hydro->target, map->target, f);
      tdpAssert(tdpPeekAtLastError());
      tdpAssert(tdpDeviceSynchronize());
    }
  }

  tdpAssert(tdpFree(flocal_d));
  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_grad_mu_fluid_kernel
 *
 *  Accumulate -phi grad mu to local force at poistion (i).
 *
 *  f_x(i) = -0.5*phi(i)*(mu(i+1) - mu(i-1)) etc
 *
 *  A number of order parameters may be present.
 *
 *****************************************************************************/

__global__ void phi_grad_mu_fluid_kernel(kernel_ctxt_t * ktx, field_t * phi,
					 fe_t * fe, hydro_t * hydro) {
  int kiterations;
  int kindex;

  assert(ktx);
  assert(phi);
  assert(hydro);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    /* NVECTOR is max size of order parameter field as need fixed array[] */
    assert(phi->nf <= NVECTOR);

    double force[3] = {0};
    double mum1[NVECTOR + 1]; /* Ugly gotcha: extra chemical potential */
    double mup1[NVECTOR + 1]; /* may exist, but not required ... */
    double phi0[NVECTOR];     /* E.g., in ternary implementation. */

    for (int n = 0; n < phi->nf; n++) {
      phi0[n] = phi->data[addr_rank1(phi->nsites, phi->nf, index, n)];
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
      int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);
      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      force[X] = 0.0;
      for (int n = 0; n < phi->nf; n++) {
	force[X] += -phi0[n]*0.5*(mup1[n] - mum1[n]);
      }
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
      int indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);
      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      force[Y] = 0.0;
      for (int n = 0; n < phi->nf; n++) {
	force[Y] += -phi0[n]*0.5*(mup1[n] - mum1[n]);
      }
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
      int indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);
      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      force[Z] = 0.0;
      for (int n = 0; n < phi->nf; n++) {
	force[Z] += -phi0[n]*0.5*(mup1[n] - mum1[n]);
      }
    }

    hydro_f_local_add(hydro, index, force);
  }
  return;
}

/*****************************************************************************
 *
 *  phi_grad_mu_solid_kernel
 *
 *  This computes and stores the force on the fluid via
 *    f_a = - phi \nabla_a mu
 *
 *  which is appropriate for the symmtric and Brazovskii
 *  free energies, This version allows a solid wall, and
 *  makes the approximation that the normal gradient of
 *  the chemical potential at the wall is zero.
 *
 *  The gradient of the chemical potential is computed as
 *    grad_x mu = 0.5*(mu(i+1) - mu(i) + mu(i) - mu(i-1)) etc
 *  which collapses to the fluid version away from any wall.
 *
 *  Ternary free energy: there are two order parameters and three
 *  chemical potentials. The force only involves the first two
 *  chemical potentials, so loops involving nf are relevant.
 *
 *  Note this case is aimed at walls, so it is assumed that all
 *  fluid sites in the domain are MAP_FLUID, and there is a check
 *  against MAP_BOUNDARY only to identify solid sites...
 *
 *****************************************************************************/

__global__ void phi_grad_mu_solid_kernel(kernel_ctxt_t * ktx, field_t * field,
					 fe_t * fe, hydro_t * hydro,
					 map_t * map) {
  int kiterations;
  int kindex;

  assert(ktx);
  assert(field);
  assert(hydro);
  assert(field->nf <= NVECTOR);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic     = kernel_coords_ic(ktx, kindex);
    int jc     = kernel_coords_jc(ktx, kindex);
    int kc     = kernel_coords_kc(ktx, kindex);
    int index0 = kernel_coords_index(ktx, ic, jc, kc);

    double force[3]      = {0};
    double phi[NVECTOR]  = {0};
    double mu[NVECTOR+1] = {0};

    field_scalar_array(field, index0, phi);
    fe->func->mu(fe, index0, mu);

    /* x-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
      int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {0};
      double mup1[NVECTOR+1] = {0};

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
      }

      for (int n1 = 0; n1 < field->nf; n1++) {
	force[X] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
      }
    }

    /* y-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
      int indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {0};
      double mup1[NVECTOR+1] = {0};

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 =0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
      }

      for (int n1 = 0; n1 < field->nf; n1++) {
	force[Y] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
      }
    }

    /* z-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
      int indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {0};
      double mup1[NVECTOR+1] = {0};

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
      }

      for (int n1 = 0; n1 < field->nf; n1++) {
	force[Z] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
      }
    }

    /* Store the force on lattice */

    hydro_f_local_add(hydro, index0, force);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_grad_mu_external_kernel
 *
 *  Accumulate local force resulting from constant external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__global__ void phi_grad_mu_external_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    double3 grad_mu, hydro_t * hydro) {
  int kiterations;
  int kindex;

  assert(ktx);
  assert(phi);
  assert(hydro);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    double force[3] = {0};
    double phi0 = phi->data[addr_rank1(phi->nsites, 1, index, 0)];

    force[X] = -phi0*grad_mu.x;
    force[Y] = -phi0*grad_mu.y;
    force[Z] = -phi0*grad_mu.z;

    hydro_f_local_add(hydro, index, force);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_grad_mu_solid_correction_kernel
 *
 *  Work out the force at each fluid site as before.
 *  This is a general version which allows general solid objects.
 *
 *  Here we accumulate the force over all sites (and the current number
 *  of fluid sites) to admit a global correction if the sum isn't zero.
 *
 *****************************************************************************/

__global__ void phi_grad_mu_correction_kernel(kernel_ctxt_t * ktx,
					      field_t * field,
					      fe_t * fe, hydro_t * hydro,
					      map_t * map, double * fcorrect) {
  int kiterations;
  int kindex;
  int tid = threadIdx.x;

  __shared__ double fv[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fx[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_PAD*TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(field);
  assert(hydro);
  assert(field->nf <= NVECTOR);

  fv[TARGET_PAD*tid] = 0.0;
  fx[TARGET_PAD*tid] = 0.0;
  fy[TARGET_PAD*tid] = 0.0;
  fz[TARGET_PAD*tid] = 0.0;

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic     = kernel_coords_ic(ktx, kindex);
    int jc     = kernel_coords_jc(ktx, kindex);
    int kc     = kernel_coords_kc(ktx, kindex);
    int index0 = kernel_coords_index(ktx, ic, jc, kc);
    int map0   = -1;

    map_status(map, index0, &map0);

    if (map0 == MAP_FLUID) {

      double force[3]      = {0};
      double phi[NVECTOR]  = {0};
      double mu[NVECTOR+1] = {0};

      field_scalar_array(field, index0, phi);
      fe->func->mu(fe, index0, mu);

      /* x-direction */
      {
	int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
	int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);
	int mapm1   = MAP_FLUID;
	int mapp1   = MAP_FLUID;

	double mum1[NVECTOR+1] = {0};
	double mup1[NVECTOR+1] = {0};

	fe->func->mu(fe, indexm1, mum1);
	fe->func->mu(fe, indexp1, mup1);

	map_status(map, indexm1, &mapm1);
	map_status(map, indexp1, &mapp1);

	if (mapm1 != MAP_FLUID) {
	  for (int n1 = 0; n1 < field->nf; n1++) {
	    mum1[n1] = mu[n1];
	  }
	}
	if (mapp1 != MAP_FLUID) {
	  for (int n1 = 0; n1 < field->nf; n1++) {
	    mup1[n1] = mu[n1];
	  }
	}

	for (int n1 = 0; n1 < field->nf; n1++) {
	  force[X] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
	}
      }

      /* y-direction */
      {
	int indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
	int indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);
	int mapm1   = MAP_FLUID;
	int mapp1   = MAP_FLUID;

	double mum1[NVECTOR+1] = {0};
	double mup1[NVECTOR+1] = {0};

	fe->func->mu(fe, indexm1, mum1);
	fe->func->mu(fe, indexp1, mup1);

	map_status(map, indexm1, &mapm1);
	map_status(map, indexp1, &mapp1);

	if (mapm1 != MAP_FLUID) {
	  for (int n1 =0; n1 < field->nf; n1++) {
	    mum1[n1] = mu[n1];
	  }
	}
	if (mapp1 != MAP_FLUID) {
	  for (int n1 = 0; n1 < field->nf; n1++) {
	    mup1[n1] = mu[n1];
	  }
	}

	for (int n1 = 0; n1 < field->nf; n1++) {
	  force[Y] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
	}
      }

      /* z-direction */
      {
	int indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
	int indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);
	int mapm1   = MAP_FLUID;
	int mapp1   = MAP_FLUID;

	double mum1[NVECTOR+1] = {0};
	double mup1[NVECTOR+1] = {0};

	fe->func->mu(fe, indexm1, mum1);
	fe->func->mu(fe, indexp1, mup1);

	map_status(map, indexm1, &mapm1);
	map_status(map, indexp1, &mapp1);

	if (mapm1 != MAP_FLUID) {
	  for (int n1 = 0; n1 < field->nf; n1++) {
	    mum1[n1] = mu[n1];
	  }
	}
	if (mapp1 != MAP_FLUID) {
	  for (int n1 = 0; n1 < field->nf; n1++) {
	    mup1[n1] = mu[n1];
	  }
	}

	for (int n1 = 0; n1 < field->nf; n1++) {
	  force[Z] -= phi[n1]*0.5*(mup1[n1] - mum1[n1]);
	}
      }

      /* Store the force on lattice */

      hydro_f_local_add(hydro, index0, force);

      fv[TARGET_PAD*tid] += 1.0;
      fx[TARGET_PAD*tid] += force[X];
      fy[TARGET_PAD*tid] += force[Y];
      fz[TARGET_PAD*tid] += force[Z];
    }
  }

  /* Accumulate the totals */

  __syncthreads();

  if (tid == 0) {
    double fvb = 0.0;
    double fxb = 0.0;
    double fyb = 0.0;
    double fzb = 0.0;

    for (int it = 0; it < blockDim.x; it++) {
      fvb += fv[TARGET_PAD*it];
      fxb += fx[TARGET_PAD*it];
      fyb += fy[TARGET_PAD*it];
      fzb += fz[TARGET_PAD*it];
    }
    tdpAtomicAddDouble(fcorrect, fvb);           /* Volume */
    tdpAtomicAddDouble(fcorrect + 1 + X, fxb);   /* force[X] */
    tdpAtomicAddDouble(fcorrect + 1 + Y, fyb);   /* force[Y] */
    tdpAtomicAddDouble(fcorrect + 1 + Z, fzb);   /* force[Z] */
  }

  return;
}

/*****************************************************************************
 *
 *  phi_grad_mu_force_kernel
 *
 *  Apply a uniform correction to local force at fluid sites.
 *
 *****************************************************************************/

__global__ void phi_grad_mu_force_kernel(kernel_ctxt_t * ktx, hydro_t * hydro,
					 map_t * map, double3 fcorrect) {
  int kiterations;
  int kindex;

  assert(ktx);
  assert(hydro);
  assert(map);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);
    int map0  = -1;

    map_status(map, index, &map0);

    if (map0 == MAP_FLUID) {
      double force[3] = {fcorrect.x, fcorrect.y, fcorrect.z};
      hydro_f_local_add(hydro, index, force);
    }
  }

  return;
}
