/*****************************************************************************
 *
 *  nernst_planck.c
 *
 *  A solution for the Nerst-Planck equation, which is the advection
 *  diffusion equation for charged species \rho_k in the presence of
 *  a potential \psi.
 *
 *  We have, in the most simple case:
 *
 *  d_t rho_k + div . (rho_k u) = div . D_k (grad rho_k + Z_k rho_k grad psi)
 *
 *  where u is the velocity field, D_k are the diffusion constants for
 *  each species k, and Z_k = (valancy_k e / k_bT) = beta valency_k e.
 *  e is the unit charge.
 *
 *  If the chemical potential is mu_k for species k, the diffusive
 *  flux may be written as
 *
 *    j_k = - D_k rho_k grad (beta mu_k)
 *
 *  with mu_k = mu_k^ideal + mu_k^ex = k_bT ln(rho_k) + valency_k e psi.
 *  (For more complex problems, there may be other terms in the chemical
 *  potential.)
 *
 *  As it is important to conserve charge, we solve in a flux form.
 *  Following Capuani, Pagonabarraga and Frenkel, J. Chem. Phys.
 *  \textbf{121} 973 (2004) we include factors to ensure that the
 *  charge densities follow a Boltzmann distribution in equilbrium.
 *
 *  This writes the flux as
 *    j_k = - D_k exp[beta mu_k^ex] grad (rho_k exp[beta mu_k^ex])
 *
 *  which we approximate at the cell faces by (e.g., for x only)
 *
 *    -D_k (1/2) { exp[-beta mu_k^ex(i)] + exp[-beta mu_k^ex(i+1)] }
 *    * { rho_k(i+1) exp[beta mu_k^ex(i+1)] - rho_k(i) exp[beta mu_k^ex(i)] }
 *
 *  We then compute the divergence of the fluxes to update via an
 *  Euler forward step. The advective fluxes (again computed at the
 *  cells faces) may be added to the diffusive fluxes to solve the
 *  whole thing. Appropraite advective fluxes may be computed via
 *  the advection.h interface.
 *
 *  Solid boundaries simply involve enforcing a no normal flux
 *  condition at the cell face.
 *
 *  The potential and charge species are available via the psi_s
 *  object.
 *
 *  A uniform external electric field may be applied; this is done
 *  by adding a contribution to the potential
 *     psi -> psi - eE.r
 *  which just appears as -eE in the calculation of grad psi.
 *
 *
 *  Edinbrugh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "advection.h"
#include "advection_bcs.h"
#include "nernst_planck.h"
#include "psi_gradients.h"

/* This needs an input switch to make it active. */
int nernst_planck_fluxes_force_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
		map_t * map, colloids_info_t * cinfo, int timestep, double ** flx);

static int nernst_planck_fluxes(psi_t * psi, fe_t * fel, double * fx,
				double * fy,
				double * fz);
static int nernst_planck_no_flux_condition(map_t * map, int nk, double * fx,
					   double * fy, double * fz);
static int nernst_planck_update(psi_t * psi, double * fe, double * fy,
				double * fz);
static int nernst_planck_fluxes_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
		map_t * map, colloids_info_t * cinfo, double ** flx);
static int nernst_planck_update_d3qx(psi_t * psi, 
				map_t * map, double ** flx);
static double max_acc; 

int np_advective_fluxes(psi_t * psi, hydro_t * hydro, double ** flux);
int np_no_flux_boundary(psi_t * psi, map_t * map, double ** flux);

int flx_noise(psi_t * psi, double ** flx, noise_t * noise,  map_t * map);
static int charge_update_sanity_check(psi_t * psi, map_t * map, double ** flx, int timestep);
static int stencil_opp_index(stencil_t * s, int c);
static int wrap_local_coord(int i, int n);

/*****************************************************************************
 *
 *  nernst_planck_driver
 *
 *  No hydrodynamics (use nernst_plancj_driver_d3qx instead).
 *  Allows for a no-flux condition between solid and fluid sites.
 *
 *****************************************************************************/

int nernst_planck_driver(psi_t * psi, fe_t * fel, map_t * map) {

  int nk;              /* Number of electrolyte species */
  int nsites;          /* Number of lattice sites */

  double * fe = NULL;
  double * fy = NULL;
  double * fz = NULL;

  psi_nk(psi, &nk);
  cs_nsites(psi->cs, &nsites);

  /* Allocate fluxes and initialise to zero */

  fe = (double*) calloc((size_t) nsites*nk, sizeof(double));
  fy = (double*) calloc((size_t) nsites*nk, sizeof(double));
  fz = (double*) calloc((size_t) nsites*nk, sizeof(double));

  if (fe == NULL) pe_fatal(psi->pe, "calloc(fe) failed\n");
  if (fy == NULL) pe_fatal(psi->pe, "calloc(fy) failed\n");
  if (fz == NULL) pe_fatal(psi->pe, "calloc(fz) failed\n");

  /* Diffusive fluxes based on six-point stencil, followed
   * by no-flux condition for solid-fluid boundaries. */

  nernst_planck_fluxes(psi, fel, fe, fy, fz);
  nernst_planck_no_flux_condition(map, nk, fe, fy, fz);

  /* Update charge distribution */

  nernst_planck_update(psi, fe, fy, fz);

  free(fz);
  free(fy);
  free(fe);

  return 0;
}


/*****************************************************************************
 *
 *  nernst_planck_fluxes
 *
 *  Compute diffusive fluxes via a simple 7-point stencil in 3d.
 *
 *****************************************************************************/

static int nernst_planck_fluxes(psi_t * psi, fe_t * fel, double * fx,
				double * fy,
				double * fz) {
  int ic, jc, kc, index;
  int nlocal[3];
  int nsites;
  int zs, ys, xs;
  int n, nk;

  double eunit, reunit;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */

  assert(psi);
  assert(fx);
  assert(fy);
  assert(fz);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	for (n = 0; n < nk; n++) {

	  fel->func->mu_solv(fel, index, n, &mu_s0);
	  mu0 = reunit*mu_s0
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index)];
	  rho0 = psi->rho->data[addr_rank1(nsites, nk, index, n)];

	  /* x-direction (between ic and ic+1) */

	  fel->func->mu_solv(fel, index + xs, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + xs)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + xs), n)]*b1;

	  fx[addr_rank1(nsites, nk, index, n)]
	    = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);

	  /* y-direction (between jc and jc+1) */

	  fel->func->mu_solv(fel, index + ys, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + ys)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + ys), n)]*b1;

	  fy[nk*index + n] = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);

	  /* z-direction (between kc and kc+1) */

	  fel->func->mu_solv(fel, index + zs, n, &mu_s1);
	  mu1 = reunit*mu_s1
	    + psi->valency[n]*psi->psi->data[addr_rank0(nsites, index + zs)];

	  b0 = exp(mu1 - mu0);
	  b1 = exp(mu1 - mu0);
	  rho1 = psi->rho->data[addr_rank1(nsites, nk, (index + zs), n)]*b1;

	  fz[addr_rank1(nsites, nk, index, n)]
	    = -psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0);
	}

	/* Next face */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_no_flux_condition
 *
 *  Use map to determine solid/fluid boudaries and set no normal flux.
 *
 *****************************************************************************/

static int nernst_planck_no_flux_condition(map_t * map, int nk, double * fx,
					   double * fy, double * fz) {

  cs_t * cs = NULL;
  int nsites = 0;
  int nlocal[3] = {0};

  assert(map);
  assert(fx);
  assert(fy);
  assert(fz);

  cs = map->cs;
  cs_nsites(cs, &nsites);
  cs_nlocal(cs, nlocal);

  for (int ic = 0; ic <= nlocal[X]; ic++) {
    for (int jc = 0; jc <= nlocal[Y]; jc++) {
      for (int kc = 0; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	int status = MAP_FLUID;
	map_status(map, index, &status);
	if (status == MAP_BOUNDARY) {
	  for (int n = 0; n < nk; n++) {
	    fx[addr_rank1(nsites, nk, index, n)] = 0.0;
	    fy[addr_rank1(nsites, nk, index, n)] = 0.0;
	    fz[addr_rank1(nsites, nk, index, n)] = 0.0;
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_update
 *
 *  Update the rho_k from the fluxes. Euler forward step.
 *
 *****************************************************************************/

static int nernst_planck_update(psi_t * psi, double * fx, double * fy,
				double * fz) {
  int ic, jc, kc, index;
  int nlocal[3];
  int zs, ys, xs;
  int n, nk;
  
  double dt;

  assert(psi);
  assert(fx);
  assert(fy);
  assert(fz);

  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	for (n = 0; n < nk; n++) {
	  psi->rho->data[addr_rank1(psi->nsites, nk, index, n)]
	    -= (+ fx[addr_rank1(psi->nsites, nk, index, n)]
		- fx[addr_rank1(psi->nsites, nk, (index-xs), n)]
		+ fy[addr_rank1(psi->nsites, nk, index, n)]
		- fy[addr_rank1(psi->nsites, nk, (index-ys), n)]
		+ fz[addr_rank1(psi->nsites, nk, index, n)]
		- fz[addr_rank1(psi->nsites, nk, (index-zs), n)])*dt;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_driver_d3qx
 *
 *  The hydro object is allowed to be NULL, in which case there is
 *  no advection.
 *
 *  The map object is allowed to be NULL, in which case no boundary
 *  condition corrections are attempted.
 *
 *****************************************************************************/

int nernst_planck_driver_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
			      map_t * map, colloids_info_t * cinfo, int timestep, noise_t * noise) {

  int nk;              /* Number of electrolyte species */
  int ia;

  double ** flx = NULL;

  psi_nk(psi, &nk);

  /* Allocate fluxes and initialise to zero */
  flx = (double **) calloc((size_t) psi->nsites*nk, sizeof(double *));
  assert(flx);
  if (flx == NULL) pe_fatal(psi->pe, "calloc(flx) failed\n");

  for (ia = 0; ia < psi->nsites*nk; ia++) {
    int nflux = psi->stencil->npoints;
    flx[ia] = (double *) calloc(nflux-1, sizeof(double));
    assert(flx[ia]);
    if (flx[ia] == NULL) pe_fatal(psi->pe, "calloc(flx[]) failed\n");
  }

  /* Add thermal fluxes */
  if (noise) flx_noise(psi, flx, noise, map);

  /* Add advective fluxes */
  if (hydro) np_advective_fluxes(psi, hydro, flx);

  /* Add diffusive fluxes */
  nernst_planck_fluxes_d3qx(psi, fe, hydro, map, cinfo, flx);

  /* Charge update Sanity Check */
  charge_update_sanity_check(psi, map, flx, timestep);
  
  /* Apply no-flux BC */
  if (map) np_no_flux_boundary(psi, map, flx);

  /* Update charges */
  nernst_planck_update_d3qx(psi, map, flx);
  

  for (ia = 0; ia < psi->nsites*nk; ia++) {
    free(flx[ia]);
  }
  free(flx);

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes_d3qx
 *
 *  Compute diffusive fluxes.
 *
 *  We assume we can accumulate the diffusive and advective fluxes separately.
 *
 *  As we compute rho(n+1) = rho(n) - div.flux in the update routine,
 *  there is an extra minus sign in the fluxes here. This conincides
 *  with the sign of the advective fluxes, if present.
 *
 *****************************************************************************/

static int nernst_planck_fluxes_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
				     map_t * map, colloids_info_t * cinfo,
				     double ** flx) {

  int ic, jc, kc; 
  int index0, index1;
  int nlocal[3];
  int n, nk; /* Number of charged species */
  int c;
  int status1;

  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */
  
  double eunit, reunit;
  double dt;

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  colloid_t * pc = NULL;

  double * __restrict__ psidata = psi->psi->data;
  double * __restrict__ rhodata = psi->rho->data;

  LB_RCS_TABLE(rcs);

  assert(psi);
  assert(fe);
  assert(fe->func->mu_solv);
  assert(flx);

  cs_nlocal(psi->cs, nlocal);

  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

	if (pc) {
	  continue;
	}
	else {
	  stencil_t * s = psi->stencil;
	  assert(s);

	  for (c = 1; c < s->npoints; c++) {

	    int8_t cx  = s->cv[c][X];
	    int8_t cy  = s->cv[c][Y];
	    int8_t cz  = s->cv[c][Z];
	    int8_t pcv = cx*cx + cy*cy + cz*cz;

	    index1 = cs_index(psi->cs, ic + cx, jc + cy, kc + cz);
	    map_status(map, index1, &status1);

	    if (status1 == MAP_FLUID) {

	      for (n = 0; n < nk; n++) {

		fe->func->mu_solv(fe, index0, n, &mu_s0);
		mu0 = reunit*mu_s0
		  + psi->valency[n]*psidata[addr_rank0(psi->nsites, index0)];
		rho0 = rhodata[addr_rank1(psi->nsites, nk, index0, n)];

		fe->func->mu_solv(fe, index1, n, &mu_s1);
		mu1 = reunit*mu_s1
		  + psi->valency[n]* psidata[addr_rank0(psi->nsites, index1)];
		b0 = exp(mu0 - mu1);
		b1 = exp(mu1 - mu0);
		rho1 = rhodata[addr_rank1(psi->nsites, nk, index1, n)]*b1;

		flx[addr_rank1(psi->nsites, nk, index0, n)][c - 1]
                  -= psi->mobility_elec[n]*kt*0.5*(1.0 + b0)*(rho1 - rho0)*rcs[pcv];
	      }
	    }
	  }
        }
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes_force_d3qx
 *
 *  Compute diffusive fluxes and link-flux force on fluid.
 *
 *  We assume we can accumulate the diffusive and advective fluxes separately.
 *
 *  As we compute rho(n+1) = rho(n) - div.flux in the update routine,
 *  there is an extra minus sign in the fluxes here. This conincides
 *  with the sign of the advective fluxes, if present.
 *
 *****************************************************************************/

int nernst_planck_fluxes_force_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro, 
				    map_t * map, colloids_info_t * cinfo, int timestep,
				    double ** flx) {

  int ic, jc, kc; 
  int index0, index1;
  int nlocal[3];
  int n, nk; /* Number of charged species */
  int nsites;
  int c;
  int status1;

  double eunit;
  double beta, rbeta;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */
  
  double rho_elec;
  double e[3];
  double flocal[4] = {0.0, 0.0, 0.0, 0.0}, fsum[4], f[3]; 
  double flxtmp[2];
  double dt;

  double kt = 0.0;
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_kt(phys, &kt);

  MPI_Comm comm;
  colloid_t * pc = NULL;

  double * __restrict__ psidata = psi->psi->data;
  double * __restrict__ rhodata = psi->rho->data;

  LB_RCS_TABLE(rcs);

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_cart_comm(psi->cs, &comm);

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  psi_beta(psi, &beta);
  psi_multistep_timestep(psi, &dt);

  rbeta = 1.0/beta;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

	f[X] = 0.0;
	f[Y] = 0.0;
	f[Z] = 0.0;

	psi_rho_elec(psi, index0, &rho_elec);

	/* Total electrostatic force on colloid */
	if (pc) {

	  psi_electric_field(psi, index0, e);

	  f[X] = rho_elec * e[X] * dt;
	  f[Y] = rho_elec * e[Y] * dt;
	  f[Z] = rho_elec * e[Z] * dt;

	  pc->force[X] += f[X];
	  pc->force[Y] += f[Y];
	  pc->force[Z] += f[Z];

	}
	else {

	  stencil_t * s = psi->stencil;
	  /* Internal electrostatic force on fluid */
	  for (c = 1; c < s->npoints; c++) {
	    int8_t cx  = s->cv[c][X];
	    int8_t cy  = s->cv[c][Y];
	    int8_t cz  = s->cv[c][Z];
	    int8_t pcv = cx*cx + cy*cy + cz*cz;

	    index1 = cs_index(psi->cs, ic + cx, jc + cy, kc + cz);
	    map_status(map, index1, &status1);

	    if (status1 == MAP_FLUID) {

	      for (n = 0; n < nk; n++) {
		fe->func->mu_solv(fe, index0, n, &mu_s0);
		mu0 = mu_s0
		  + psi->valency[n]*eunit*psidata[addr_rank0(nsites, index0)];
		rho0 = rhodata[addr_rank1(nsites, nk, index0, n)];

		fe->func->mu_solv(fe, index1, n, &mu_s1);
		mu1 = mu_s1
		  + psi->valency[n]*eunit*psidata[addr_rank0(nsites, index1)];
		b0 = exp(-beta*(mu1 - mu0));
		b1 = exp(+beta*(mu1 - mu0));
		rho1 = rhodata[addr_rank1(nsites, nk, index1, n)]*b1;

		flxtmp[0] = - 0.5*(1.0 + b0)*(rho1 - rho0)*rcs[pcv];

		/* Diffusive flux accumulated */
		flx[addr_rank1(nsites, nk, index0, n)][c - 1] += psi->mobility_elec[n]*kt*flxtmp[0];

		/* Force, including ideal gas part in chemical potential */
		f[X] -= s->wgradients[c]*cx*flxtmp[0]*rbeta;
		f[Y] -= s->wgradients[c]*cy*flxtmp[0]*rbeta;
		f[Z] -= s->wgradients[c]*cz*flxtmp[0]*rbeta;
	      }
	    }
	  }

	  /* Electrostatic force in external field */

	  f[X] *= dt;
	  f[Y] *= dt;
	  f[Z] *= dt;

	  /* Count number of fluid sites */
	  flocal[3] += 1.0;

          if (hydro) hydro_f_local_add(hydro, index0, f);

        }   

	/* Accumulate contribution to total force on system */ 
	flocal[X] += f[X];
	flocal[Y] += f[Y];
	flocal[Z] += f[Z];

      }
    }
  }

  /* On fluid sites apply correction for momentum conservation */

  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);
 
  fsum[X] /= fsum[3];
  fsum[Y] /= fsum[3];
  fsum[Z] /= fsum[3];

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(psi->cs, ic, jc, kc);
        colloids_info_map(cinfo, index0, &pc);

	if (pc) continue;

	f[X] = -fsum[X];
	f[Y] = -fsum[Y];
	f[Z] = -fsum[Z];

	if (hydro) hydro_f_local_add(hydro, index0, f);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_update_d3qx
 *
 *  Update the rho_k from the fluxes (D3QX stencil). Euler forward step.
 *
 *****************************************************************************/

static int nernst_planck_update_d3qx(psi_t * psi, map_t * map, double ** flx) {

  int ic, jc, kc, index;
  int nsites;
  int nlocal[3];
  int n, nk;
  int c;
  int status;
  double acc, maxacc=0.0;
  double dt;

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);

  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);
        map_status(map, index, &status);

        if (status == MAP_FLUID) {
	  stencil_t * s = psi->stencil;
	  for (n = 0; n < nk; n++) {

	    acc = 0.0;
	    for (c = 1; c < s->npoints; c++) {
	      psi->rho->data[addr_rank1(nsites, nk, index, n)]
		-= flx[addr_rank1(nsites, nk, index, n)][c - 1] * dt;
	      acc += fabs(flx[addr_rank1(nsites, nk, index, n)][c - 1] * dt);
	    }

	    acc /= fabs(psi->rho->data[addr_rank1(nsites, nk, index, n)]);
	    if (maxacc < acc) maxacc = acc; 
	  }
	}

      }
    }
  }

  nernst_planck_maxacc_set(maxacc);

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_maxacc_set
 *
 *  Setter function for the local maximal accuracy in the Nernst-Planck 
 *  equation. This is defined as the absolut value of the ratio of the change 
 *  during one fractional LB timestep (multistep dt) and the charge 
 *  density itself.
 *  
 *****************************************************************************/

int nernst_planck_maxacc_set(double acc) {
  max_acc = acc;
  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_maxacc
 *
 *  Getter function for the local maximal accuracy 
 *  in the Nernst-Planck equation.
 *
 *****************************************************************************/

int nernst_planck_maxacc(double * acc) {
  * acc = max_acc;
  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_adjust_multistep
 *
 *  
 *****************************************************************************/

int nernst_planck_adjust_multistep(psi_t * psi) {

  double maxacc_local[1], maxacc[1], diffacc; /* actual and preset value of diffusive accuracy */
  double diff, diffmax=0.0;                   /* diffusivity of species and maximal value      */ 
  int n, nk, multisteps;
  MPI_Comm comm;

  psi_diffacc(psi, &diffacc);

  /* Take local maximum and reduce for global maximum */
  nernst_planck_maxacc(&maxacc_local[0]);
  cs_cart_comm(psi->cs, &comm);
  MPI_Allreduce(maxacc_local, maxacc, 1, MPI_DOUBLE, MPI_MAX, comm);

  /* Compare maximal accuracy with preset value for */ 
  /*   diffusion and adjust number of multisteps    */

  /* Increase no. of multisteps */
  if (* maxacc > diffacc && diffacc > 0.0) {
    psi_multisteps(psi, &multisteps);
    multisteps *= 2;
    psi->multisteps = multisteps;
    pe_info(psi->pe, "\nMaxacc > diffacc: changing no. of multisteps to %d\n",
	    multisteps);
  }    

  /* Reduce no. of multisteps */
  /* The factor 0.1 prevents too frequent changes. */
  if (* maxacc < 0.1*diffacc && diffacc > 0.0) {
  
    psi_multisteps(psi, &multisteps);
    psi_nk(psi, &nk);

    for (n = 0; n < nk; n++) {
	psi_diffusivity(psi, n, &diff);
	if (diff > diffmax) diffmax = diff;
    }

    /* Only reduce if sanity criteria fulfilled */  
    if (multisteps > 1 && diffmax/multisteps < 0.05) { 
      multisteps *= 0.5; 
      psi->multisteps = multisteps;
      pe_info(psi->pe, "\nMaxacc << diffacc: changing no. of multisteps to %d\n", multisteps);
    }    

  }    

  return 0;
} 

/*****************************************************************************
 *
 *  Thermal Noise
 *
 *****************************************************************************/
int flx_noise(psi_t * psi, double ** flx, noise_t * noise,  map_t * map) {

  int nlocal[3] = {0};
  int ntotal[3] = {0};
  int noffset[3] = {0};
  int index0, index1;
  int vaddr0_X, vaddr0_Y, vaddr0_Z, vaddr1_X, vaddr1_Y, vaddr1_Z;
  double rho0, rho1, rho_face;
  double flux;
  int n, nk; /* Number of charged species */
  double kt = 0.0;
  double mobility_elec = 0.0;
  int status0, status1;

  cs_t * cs = NULL;
  stencil_t * s = NULL;
  physics_t * phys = NULL;
  static field_t * var = NULL;

  double * __restrict__ rho = NULL;

  cs = psi->cs;
  s  = psi->stencil;
  rho = psi->rho->data;

  assert(psi);
  assert(flx);
  assert(noise);
  assert(cs);
  assert(s);
  assert(rho);

  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);
  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_nk(psi, &nk);

  if (var == NULL) {
    field_options_t opts = field_options_ndata_nhalo(3, 3);
    field_create(psi->pe, psi->cs, psi->rho->le, "noise-var", &opts, &var);
  }

  for (n = 0; n < nk; n++) {
    
    mobility_elec = psi->mobility_elec[n];
    phi_ch_var_flux_driver(var, noise, mobility_elec, kt);

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {

          index0 = cs_index(cs, ic, jc, kc);
          vaddr0_X = addr_rank1(var->nsites, 3, index0, X);
          vaddr0_Y = addr_rank1(var->nsites, 3, index0, Y);
          vaddr0_Z = addr_rank1(var->nsites, 3, index0, Z);

          map_status(map, index0, &status0);

          for (int p = 1; p < s->npoints; p++) {

            int8_t cx = s->cv[p][X];
            int8_t cy = s->cv[p][Y];
            int8_t cz = s->cv[p][Z];

            index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
            vaddr1_X = addr_rank1(var->nsites, 3, index1, X);
            vaddr1_Y = addr_rank1(var->nsites, 3, index1, Y);
            vaddr1_Z = addr_rank1(var->nsites, 3, index1, Z);
            map_status(map, index1, &status1);

            rho0 = rho[addr_rank1(psi->nsites, psi->nk, index0, n)];
            rho1 = rho[addr_rank1(psi->nsites, psi->nk, index1, n)];
            rho_face = 0.5 * (rho0 + rho1); 
            flux = 0.0;
            if (rho_face <= 0.0) continue;

            if (status1 == MAP_FLUID && status0 == MAP_FLUID) {
              // works for 7 point stencil
              if (cx == 1 || cy == 1 || cz == 1){
                flux = sqrt(rho_face) * (var->data[vaddr0_X] * cx + var->data[vaddr0_Y] * cy+ var->data[vaddr0_Z] * cz);
              }

              else if (cx == -1 || cy == -1 || cz == -1){
                flux = sqrt(rho_face) * (var->data[vaddr1_X] * cx + var->data[vaddr1_Y] * cy+ var->data[vaddr1_Z] * cz);
              }
            
              flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] += flux;

              
            }

          }
        }
      }
    }
  }
  
  return 0;
}


/*****************************************************************************
 *
 *  np_advective_fluxes
 *
 *  'Centred difference' advective fluxes for the char densities rho.
 *
 *  Symmetric two-point stencil.
 *
 *****************************************************************************/

int np_advective_fluxes(psi_t * psi, hydro_t * hydro, double ** flx) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  double * __restrict__ rho = psi->rho->data;

  assert(psi);
  assert(hydro);
  assert(flx);

  cs = psi->cs;
  s = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index0 = cs_index(cs, ic, jc, kc);
	double u0[3] = {0};
	hydro_u(hydro, index0, u0);

        for (int p = 1; p < s->npoints; p++) {

	  int8_t cx = s->cv[p][X];
	  int8_t cy = s->cv[p][Y];
	  int8_t cz = s->cv[p][Z];
	  int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
	  double u1[3] = {0};
	  double u = 0.0;
	  hydro_u(hydro, index1, u1);

	  u = 0.5*((u0[X]+u1[X])*cx + (u0[Y]+u1[Y])*cy + (u0[Z]+u1[Z])*cz);

	  for (int n = 0; n < psi->nk; n++) {
	    double rho0 = rho[addr_rank1(psi->nsites, psi->nk, index0, n)];
	    double rho1 = rho[addr_rank1(psi->nsites, psi->nk, index1, n)];
	    double flux = u*0.5*(rho0 + rho1);
	    flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] += flux;
	  }
	}
	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux_d3qx
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int np_no_flux_boundary(psi_t * psi, map_t * map, double ** flx) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  assert(psi);
  assert(map);
  assert(flx);

  cs = psi->cs;
  s  = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index0 = cs_index(cs, ic, jc, kc);
	int mask0  = 1;
	int status = MAP_BOUNDARY;
	map_status(map, index0, &status);
	mask0 = (status == MAP_FLUID);

	for (int p = 1; p < s->npoints; p++) {

	  int8_t cx = s->cv[p][X];
	  int8_t cy = s->cv[p][Y];
	  int8_t cz = s->cv[p][Z];
	  int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
	  int mask = 1;

	  map_status(map, index1, &status);
	  mask = (status == MAP_FLUID);
	  mask = mask*mask0;

	  for (int n = 0;  n < psi->nk; n++) {
	    flx[addr_rank1(psi->nsites, psi->nk, index0, n)][p-1] *= mask;
	  }
	}
      }
    }
  }

  return 0;
}

static int wrap_local_coord(int i, int n) {
  if (i < 1) return n;
  if (i > n) return 1;
  return i;
}

static int stencil_opp_index(stencil_t * s, int c) {

  int cx = s->cv[c][X];
  int cy = s->cv[c][Y];
  int cz = s->cv[c][Z];

  for (int cp = 1; cp < s->npoints; cp++) {
    if (s->cv[cp][X] == -cx &&
        s->cv[cp][Y] == -cy &&
        s->cv[cp][Z] == -cz) {
      return cp;
    }
  }

  return -1;
}


static int charge_update_sanity_check(psi_t * psi, map_t * map,
                                      double ** flx, int timestep) {

  int nsites;
  int nlocal[3];
  int nk;
  double dt;

  int counter = 0;
  int counter_links = 0;

  assert(psi);
  assert(flx);

  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  psi_nk(psi, &nk);
  psi_multistep_timestep(psi, &dt);

  stencil_t * s = psi->stencil;
  assert(s);

  int nodi = nlocal[X] * nlocal[Y] * nlocal[Z];

  /*
   * Iterate a few times because removing a link flux may change
   * the predicted rho value in neighbouring cells.
   */
  int max_pass = 4;

  for (int pass = 0; pass < max_pass; pass++) {

    int changed_this_pass = 0;

    for (int ic = 1; ic <= nlocal[X]; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
        for (int kc = 1; kc <= nlocal[Z]; kc++) {

          int index0 = cs_index(psi->cs, ic, jc, kc);

          int status0 = MAP_FLUID;
          if (map) map_status(map, index0, &status0);

          if (status0 != MAP_FLUID) continue;

          for (int n = 0; n < nk; n++) {

            int ia0 = addr_rank1(nsites, nk, index0, n);

            double rho_new = psi->rho->data[ia0];

            for (int c = 1; c < s->npoints; c++) {
              rho_new -= flx[ia0][c - 1] * dt;
            }

            /*
             * If rho_new remains finite and non-negative,
             * no correction is required.
             */
            if (isfinite(rho_new) && rho_new >= 0.0) {
              continue;
            }

            /*
             * At this point the cell is at risk of becoming negative.
             *
             * Do not remove all fluxes.
             * Remove only outgoing fluxes, i.e. those with flx > 0.
             *
             * For each removed link, also remove the corresponding
             * opposite link in the neighbouring cell.
             */
            counter++;
            changed_this_pass = 1;

            for (int c = 1; c < s->npoints; c++) {

              double f = flx[ia0][c - 1];

              /*
               * f > 0 means that this cell loses mass:
               *
               * rho_i <- rho_i - dt*f
               */
              if (!isfinite(f) || f > 0.0) {

                int cx = s->cv[c][X];
                int cy = s->cv[c][Y];
                int cz = s->cv[c][Z];

                int ic1 = wrap_local_coord(ic + cx, nlocal[X]);
                int jc1 = wrap_local_coord(jc + cy, nlocal[Y]);
                int kc1 = wrap_local_coord(kc + cz, nlocal[Z]);

                int index1 = cs_index(psi->cs, ic1, jc1, kc1);

                int status1 = MAP_FLUID;
                if (map) map_status(map, index1, &status1);

                int copp = stencil_opp_index(s, c);

                /*
                 * Remove the local flux.
                 */
                flx[ia0][c - 1] = 0.0;

                /*
                 * Also remove the opposite flux in the neighbouring cell.
                 * This preserves:
                 *
                 * flx[i][c] = -flx[j][c_opp]
                 */
                if (status1 == MAP_FLUID && copp > 0) {
                  int ia1 = addr_rank1(nsites, nk, index1, n);
                  flx[ia1][copp - 1] = 0.0;
                }

                counter_links++;
              }
            }
          }
        }
      }
    }

    if (changed_this_pass == 0) break;
  }

  double counter_perc =
      (double) counter / ((double) nk * (double) nodi);

  if (counter_perc > 0.001) {
    printf("Conservative flux eraser calls %lf at ts %d; erased links %d\n",
           counter_perc, timestep, counter_links);
  }

  return 0;
}
