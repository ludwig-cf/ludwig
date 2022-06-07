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
 *  (c) 2021 The University of Edinburgh
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
					 fe_t * fe, hydro_t * hydro,
					field_t * subgrid_potential);
__global__ void phi_grad_mu_solid_kernel(kernel_ctxt_t * ktx, field_t * phi,
					 fe_t * fe, hydro_t * hydro,
					 map_t * map, field_t * subgrid_potential);
__global__ void phi_grad_mu_external_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    double3 grad_mu, hydro_t * hydro);
__global__ void phi_grad_mu_external_ll_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    double3 grad_mu_phi, double3 grad_mu_psi,
						 hydro_t * hydro);


/*****************************************************************************
 *
 *  phi_grad_mu_fluid
 *
 *  Driver for fluid only for given chemical potential (from abstract
 *  free energy description).
 *
 *****************************************************************************/

__host__ int phi_grad_mu_fluid(cs_t * cs, field_t * phi, fe_t * fe,
			       hydro_t * hydro, field_t * subgrid_potential) {
  int nlocal[3];
  dim3 nblk, ntpb;

  fe_t * fe_target;
  field_t * sf_target;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  
  assert(cs);
  assert(phi);
  assert(hydro);

  cs_nlocal(cs, nlocal);
  fe->func->target(fe, &fe_target);
  sf_target = subgrid_potential->target;

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_grad_mu_fluid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, fe_target, hydro->target,
		sf_target);

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
			       hydro_t * hydro, map_t * map, 
				field_t * subgrid_potential) {
  int nlocal[3];
  dim3 nblk, ntpb;

  fe_t * fe_target;
  field_t * sf_target;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);
  assert(map);

  cs_nlocal(cs, nlocal);
  fe->func->target(fe, &fe_target);
  sf_target = subgrid_potential->target;

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_grad_mu_solid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, phi->target, fe_target, hydro->target,
		  map->target, sf_target);

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
  double3 grad_mu = {};

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);

  cs_nlocal(cs, nlocal);

  {
    physics_t * phys = NULL;
    double mu[3] = {};

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
 *  phi_grad_mu_external_ll
 *
 *  Driver to accumulate the force originating from external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__host__ int phi_grad_mu_external_ll(cs_t * cs, field_t * phi, hydro_t * hydro) {

  int nlocal[3];
  int is_grad_mu_phi = 0;     /* Short circuit the kernel if not required. */
  int is_grad_mu_psi = 0;     /* Short circuit the kernel if not required. */
  dim3 nblk, ntpb;
  double3 grad_mu_phi = {};
  double3 grad_mu_psi = {};

  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(cs);
  assert(phi);
  assert(hydro);

  cs_nlocal(cs, nlocal);

  {
    physics_t * phys = NULL;
    double mu_phi[3] = {};
    double mu_psi[3] = {};

    physics_ref(&phys);
    physics_grad_mu_phi(phys, mu_phi);
    physics_grad_mu_psi(phys, mu_psi);

    grad_mu_phi.x = mu_phi[X];
    grad_mu_phi.y = mu_phi[Y];
    grad_mu_phi.z = mu_phi[Z];

    grad_mu_psi.x = mu_psi[X];
    grad_mu_psi.y = mu_psi[Y];
    grad_mu_psi.z = mu_psi[Z];
 
    is_grad_mu_phi = (mu_phi[X] != 0.0 || mu_phi[Y] != 0.0 || mu_phi[Z] != 0.0);
    is_grad_mu_psi = (mu_psi[X] != 0.0 || mu_psi[Y] != 0.0 || mu_psi[Z] != 0.0);
  }

  /* We may need to revisit the external chemical potential if required
   * for more than one order parameter. */

  if ((is_grad_mu_phi || is_grad_mu_psi) && phi->nf == 2) {

    limits.imin = 1; limits.imax = nlocal[X];
    limits.jmin = 1; limits.jmax = nlocal[Y];
    limits.kmin = 1; limits.kmax = nlocal[Z];

    kernel_ctxt_create(cs, 1, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(phi_grad_mu_external_ll_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, phi->target, grad_mu_phi, grad_mu_psi, hydro->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

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
					 fe_t * fe, hydro_t * hydro, 
					field_t * subgrid_potential) {
  int kiterations;
  int kindex;
  assert(ktx);
  assert(phi);
  assert(hydro);

/* -----> For book-keeping */
  double globalforce[3];
  double localforce[3] = {0,0,0};
  int writefreq = 10;

  physics_t * phys;
  FILE * fp;

  physics_ref(&phys);
  int timestep = physics_control_timestep(phys);
  MPI_Comm comm;
  cs_cart_comm(hydro->cs, &comm);
/* <----- */

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic    = kernel_coords_ic(ktx, kindex);
    int jc    = kernel_coords_jc(ktx, kindex);
    int kc    = kernel_coords_kc(ktx, kindex);
    int index = kernel_coords_index(ktx, ic, jc, kc);

    /* NVECTOR is max size of order parameter field as need fixed array[] */
    assert(phi->nf <= NVECTOR);

    double force[3] = {};
    double mum1[NVECTOR + 1]; /* Ugly gotcha: extra chemical potential */
    double mup1[NVECTOR + 1]; /* may exist, but not required ... */
    double um1;
    double up1;
    double phi0[NVECTOR];     /* E.g., in ternary implementation. */

    for (int n = 0; n < phi->nf; n++) {
      phi0[n] = phi->data[addr_rank1(phi->nsites, phi->nf, index, n)];
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
      int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      force[X] = 0.0;
      force[X] += -phi0[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[X] += -phi0[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[X] -= phi0[0]*0.5*(up1 - um1);
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
      int indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      force[Y] = 0.0;
      force[Y] += -phi0[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[Y] += -phi0[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[Y] -= phi0[0]*0.5*(up1 - um1);
    }

    {
      int indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
      int indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      force[Z] = 0.0;
      force[Z] += -phi0[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[Z] += -phi0[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[Z] -= phi0[0]*0.5*(up1 - um1);
    }

    hydro_f_local_add(hydro, index, force);
  }

  if (timestep % writefreq == 0) {
    MPI_Allreduce(localforce, globalforce, 3, MPI_DOUBLE, MPI_SUM, comm);
    fp = fopen("TOT_INTERACT_FORCE_FLUID.txt","a");
    fprintf(fp, "%14.7e, %14.7e, %14.7e\n", globalforce[X], globalforce[Y], 
					globalforce[Z]);
    fclose(fp);
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
 *****************************************************************************/

__global__ void phi_grad_mu_solid_kernel(kernel_ctxt_t * ktx, field_t * field,
					 fe_t * fe, hydro_t * hydro,
					 map_t * map, field_t * subgrid_potential) {
  int kiterations;
  int kindex;
  assert(ktx);
  assert(field);
  assert(hydro);
  assert(field->nf <= NVECTOR);

/* -----> For book-keeping */
  double globalforce[3];
  double localforce[3] = {0,0,0};
  int writefreq = 10;

  physics_t * phys;
  FILE * fp;

  physics_ref(&phys);
  int timestep = physics_control_timestep(phys);
  MPI_Comm comm;
  cs_cart_comm(hydro->cs, &comm);
/* <----- */

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic     = kernel_coords_ic(ktx, kindex);
    int jc     = kernel_coords_jc(ktx, kindex);
    int kc     = kernel_coords_kc(ktx, kindex);
    int index0 = kernel_coords_index(ktx, ic, jc, kc);

    double force[3]      = {};
    double phi[NVECTOR]  = {};
    double mu[NVECTOR+1] = {};
    double um1;
    double up1;
    double u;

    field_scalar_array(field, index0, phi);
    fe->func->mu(fe, index0, mu);
    field_scalar(subgrid_potential, index0, &u);

    /* x-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
      int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {};
      double mup1[NVECTOR+1] = {};
      double um1;
      double up1;

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
	um1 = u;
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
	up1 = u;
      }

      force[X] -= phi[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[X] -= phi[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[X] -= phi[0]*0.5*(up1 - um1);

    }

    /* y-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
      int indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {};
      double mup1[NVECTOR+1] = {};
      double um1;
      double up1;

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 =0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
	um1 = u;
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
	up1 = u;
      }

      force[Y] -= phi[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[Y] -= phi[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[Y] -= phi[0]*0.5*(up1 - um1);
      //if (timestep % writefreq == 0 && up1-um1 != 0.0) printf("fluid %14.7e\n", localforce[Y]);
    }

    /* z-direction */
    {
      int indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
      int indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);
      int mapm1   = MAP_FLUID;
      int mapp1   = MAP_FLUID;

      double mum1[NVECTOR+1] = {};
      double mup1[NVECTOR+1] = {};
      double um1;
      double up1;

      fe->func->mu(fe, indexm1, mum1);
      fe->func->mu(fe, indexp1, mup1);
      field_scalar(subgrid_potential, indexm1, &um1);
      field_scalar(subgrid_potential, indexp1, &up1);

      map_status(map, indexm1, &mapm1);
      map_status(map, indexp1, &mapp1);

      if (mapm1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mum1[n1] = mu[n1];
	}
	um1 = u;
      }
      if (mapp1 == MAP_BOUNDARY) {
	for (int n1 = 0; n1 < field->nf; n1++) {
	  mup1[n1] = mu[n1];
	}
	up1 = u;
      }

      force[Z] -= phi[0]*0.5*(mup1[0] - mum1[0] + up1 - um1);
      force[Z] -= phi[1]*0.5*(mup1[1] - mum1[1]);
      if (timestep % writefreq == 0) localforce[Z] -= phi[0]*0.5*(up1 - um1);

    }

    /* Store the force on lattice */

    hydro_f_local_add(hydro, index0, force);
  }

  if (timestep % writefreq == 0) {
    MPI_Allreduce(localforce, globalforce, 3, MPI_DOUBLE, MPI_SUM, comm);
    fp = fopen("TOT_INTERACT_FORCE_FLUID.txt","a");
    fprintf(fp, "%14.7e, %14.7e, %14.7e\n", globalforce[X], globalforce[Y], 
					globalforce[Z]);
    fclose(fp);
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

    double force[3] = {};
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
 *  phi_grad_mu_external_ll_kernel
 *
 *  Accumulate local force resulting from constant external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__global__ void phi_grad_mu_external_ll_kernel(kernel_ctxt_t * ktx, field_t * phi,
					    double3 grad_mu_phi, double3 grad_mu_psi, 
						hydro_t * hydro) {
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

    double force[3] = {};
    double phi0 = phi->data[addr_rank1(phi->nsites, 2, index, 0)];
    double psi0 = phi->data[addr_rank1(phi->nsites, 2, index, 1)];

    force[X] = -phi0*grad_mu_phi.x - psi0*grad_mu_psi.x;
    force[Y] = -phi0*grad_mu_phi.y - psi0*grad_mu_psi.y;
    force[Z] = -phi0*grad_mu_phi.z - psi0*grad_mu_psi.z;

    hydro_f_local_add(hydro, index, force);
  }

  return;
}
