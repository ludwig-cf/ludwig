/*****************************************************************************
 *
 *  phi_cahn_hilliard.c
 *
 *  The time evolution of the order parameter phi is described
 *  by the Cahn Hilliard equation
 *
 *     d_t phi + div (u phi - M grad mu - \hat{xi}) = 0.
 *
 *  The equation is solved here via finite difference. The velocity
 *  field u is assumed known from the hydrodynamic sector. M is the
 *  order parameter lambda. The chemical potential mu is set via
 *  the choice of free energy.
 *
 *  Random fluxes \hat{xi} can be included if required using the
 *  lattice noise generator with variance 2 M kT. The implementation
 *  is (broadly) based on that of Sumesh PT et al, Phys Rev E 84
 *  046709 (2011).
 *
 *  The important thing for the noise is that an expanded stencil
 *  is required for the diffusive fluxes, here via phi_ch_flux_mu2().
 *
 *  Lees-Edwards planes are allowed (but not with noise, at present).
 *  This requires fixes at the plane boudaries to get consistent
 *  fluxes.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributions:
 *  Thanks to Markus Gross, who helped to validate the noise implementation.
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "field_s.h"
#include "physics.h"
#include "advection_s.h"
#include "advection_bcs.h"
#include "util_sum.h"
#include "heat_equation.h"

static int heq_flux_mu1(heq_t * heq, fe_t * fes);
static int heq_flux_mu2(heq_t * heq, fe_t * fes);
static int heq_update_forward_step(heq_t * heq, field_t * temperaturef);
static int heq_flux_mu_ext(heq_t * heq);

static int heq_update_conserve(heq_t * heq, field_t * temperaturef);

static int heq_le_fix_fluxes(heq_t * heq, int nf);
static int heq_le_fix_fluxes_parallel(heq_t * heq, int nf);
static int heq_random_flux(heq_t * heq, noise_t * noise);

static int heq_subtract_sum_temperature_after_forward_step(heq_t * heq,
						      field_t * temperaturef,
						      map_t * map);
/* Utility container */

typedef struct heq_kernel_s heq_kernel_t;
struct heq_kernel_s {
  double lambda;      /* Mobility */
  double gradmu_ex[3];  /* External chemical potential gradient */
};

/* Utility container for corrections to order parameter conservation */

typedef struct temperature_correct_s temperature_correct_t;
struct temperature_correct_s {
  int initial;   /* Are we at t > 0? */
  int nfluid;    /* Fluid volume */
/* TODO: relevance for Temperature to be discussed */
  double T0;   /* Sum phi at t = 0 */
  double temperature;    /* Sum current. */
};

__global__ void heq_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, fe_t * fe,
				       advflux_t * flux, double lambda);
__global__ void heq_flux_mu_ext_kernel(kernel_ctxt_t * ktx,
					  lees_edw_t * le, advflux_t * flux,
					  heq_kernel_t heq_kernel);


/* TODO: relevance for Temperature to be discussed */
__global__ void heq_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux,
				  int ys, double wz);
__global__ void heq_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				   field_t * field, advflux_t * flux,
				   field_t * csum, int ys, double wz);

/*****************************************************************************
 *
 *  heq_create
 *
 *****************************************************************************/

__host__ int heq_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			   heq_info_t * options,
			   heq_t ** heq) {

  heq_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(le);
  assert(options);
  assert(heq);

  obj = (heq_t *) calloc(1, sizeof(heq_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(heq_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->info = *options;
  advflux_le_create(pe, cs, le, 1, &obj->flux);

  if (obj->info.conserve) {
    field_create(pe, cs, 1, "compensated sum", &obj->csum);
    field_init(obj->csum, 0, NULL);
  }

  pe_retain(pe);
  lees_edw_retain(le);

  *heq = obj;

  return 0;
}

/*****************************************************************************
 *
 *  heq_free
 *
 *****************************************************************************/

__host__ int heq_free(heq_t * heq) {

  assert(heq);

  lees_edw_free(heq->le);
  pe_free(heq->pe);

  if (heq->csum) field_free(heq->csum);
  advflux_free(heq->flux);
  free(heq);
  
  return 0;
}

/*****************************************************************************
 *
 *  heat_equation
 *
 *  Compute the fluxes (advective/diffusive) and compute the update
 *  to the order parameter field phi.
 *
 *  Conservation is ensured by face-flux uniqueness. However, in the
 *  x-direction, the fluxes at the east face and west face of a given
 *  cell must be handled spearately to take account of Lees Edwards
 *  boundaries.
 *
 *  hydro is allowed to be NULL, in which case the dynamics is
 *  just relaxational (no velocity field).
 *
 *  map is also allowed to be NULL, in which case there is no
 *  check for for the surface no flux condition. (This still
 *  may be relevant for diffusive fluxes only, so does not
 *  depend on hydrodynamics.)
 *
 *  The noise_t structure controls random fluxes, if required;
 *  it can be NULL in which case no noise.
 *
 *  TODO:
 *  advection_bcs_wall() may be required if 3rd or 4th order
 *  advection is used in the presence of solid. Not present
 *  at the moment.
 *
 *****************************************************************************/

int heat_equation(heq_t * heq, fe_t * fe, field_t * temperature,
		      hydro_t * hydro, map_t * map, noise_t * noise) {
  int nf;
  int noise_temperature = 0;

  assert(heq);
  assert(fe);
  assert(temperature);
  /* TODO: Add NOISE_TEMPERATURE in noise.c ? */
  if (noise) noise_present(noise, NOISE_PHI, &noise_temperature);

  field_nf(temperature, &nf);
  assert(nf == 1); 

  /* Compute any advective fluxes first, then accumulate diffusive
   * and random fluxes. */

  if (hydro) {
    hydro_u_halo(hydro); /* Reposition to main to prevent repeat */
    hydro_lees_edwards(hydro); /* Repoistion to main ditto */ 
    advection_x(heq->flux, hydro, temperature);
  }
  else {
    /* Remember to initialise fluxes to zero for this step */
    advflux_zero(heq->flux);
  }

  if (noise_temperature) {
    heq_flux_mu2(heq, fe);
    heq_random_flux(heq, noise);
  }
  else {
    heq_flux_mu1(heq, fe); 
  }

  /* External chemical potential gradient (could switch out if zero)
     TODO: Could be used to have a thermostat  */

  heq_flux_mu_ext(heq);

  /* No flux boundaries (diffusive fluxes, and hydrodynamic, if present)
     TODO: Think about = flux boundaries  */

  if (map) advection_bcs_no_normal_flux(nf, heq->flux, map);

  heq_le_fix_fluxes(heq, nf);

  /* TODO REPLACE 1/2 WITH MEANINGFUL SYMBOLS */
  /* TODO: relevance for tmperature to be discussed. For now set to 0   */
  heq->info.conserve = 0;
  if (heq->info.conserve == 1) {
    heq_update_conserve(heq, temperature);
  }
  else {
    heq_update_forward_step(heq, temperature);
    if (heq->info.conserve == 2) {
      heq_subtract_sum_temperature_after_forward_step(heq, temperature, map);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  heq_flux_mu1
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

static int heq_flux_mu1(heq_t * heq, fe_t * fe) {

  int nlocal[3];
  double lambda;
  dim3 nblk, ntpb;
  kernel_info_t limits;

  fe_t * fetarget = NULL;
  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);
  assert(fe);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &letarget);
  fe->func->target(fe, &fetarget);

  physics_ref(&phys);
  physics_lambda(phys, &lambda);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_flux_mu1_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, fetarget, heq->flux->target, lambda);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  heq_flux_mu1_kernel
 *
 *  Unvectorised kernel.
 *
 *  Accumulate [add to a previously computed advective flux] the
 *  'diffusive' contribution related to the chemical potential. It's
 *  computed everywhere regardless of fluid/solid status.
 *
 *  This is a two point stencil the in the chemical potential,
 *  and the lambda is constant.
 *
 *****************************************************************************/

__global__ void heq_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, fe_t * fe,
				       advflux_t * flux, double lambda) {
  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(le);
  assert(fe);
  assert(fe->func->mu);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index0, index1;
    int icm1, icp1;
    double mu0, mu1;
    /* Here mu0 and mu1 are taken to be the values of the temperature at a given node... which is stored in fed. Not ideal but only way to access temperature from the abstract free energy structure without changing the vtable */

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);

    index0 = lees_edw_index(le, ic, jc, kc);

    fe->func->fed(fe, index0, &mu0);

    /* x-direction (between ic-1 and ic) */

    index1 = lees_edw_index(le, icm1, jc, kc);
    fe->func->fed(fe, index1, &mu1);
    flux->fw[addr_rank0(flux->nsite, index0)] -= lambda*(mu0 - mu1);

    /* ...and between ic and ic+1 */

    index1 = lees_edw_index(le, icp1, jc, kc);
    fe->func->fed(fe, index1, &mu1);
    flux->fe[addr_rank0(flux->nsite, index0)] -= lambda*(mu1 - mu0);

    /* y direction */

    index1 = lees_edw_index(le, ic, jc+1, kc);
    fe->func->fed(fe, index1, &mu1);
    flux->fy[addr_rank0(flux->nsite, index0)] -= lambda*(mu1 - mu0);

    /* z direction */

    index1 = lees_edw_index(le, ic, jc, kc+1);
    fe->func->fed(fe, index1, &mu1);
    flux->fz[addr_rank0(flux->nsite, index0)] -= lambda*(mu1 - mu0);

    /* Next site */
  }

  return;
}

/* TODO: Relevance for temperature to be discussed */
/*****************************************************************************
 *
 *  heq_flux_mu2
 *
 *  Accumulate [add to previously computed advective fluxes]
 *  diffusive fluxes related to the lambda.
 *
 *  This version is based on Sumesh et al to allow correct
 *  treatment of noise. The fluxes are calculated via
 *
 *  grad_x mu = 0.5*(mu(i+1) - mu(i-1)) etc
 *  flux_x(x + 1/2) = -0.5*M*(grad_x mu(i) + grad_x mu(i+1)) etc
 *
 *  In contrast to Sumesh et al., we don't have 'diagonal' fluxes
 *  yet. There are also no Lees Edwards planes yet.
 *
 *****************************************************************************/

static int heq_flux_mu2(heq_t * heq, fe_t * fesymm_oft) {

  int nhalo;
  int nlocal[3];
  int ic, jc, kc;
  int index0;
  int xs, ys, zs;
  double mum2, mum1, mu00, mup1, mup2;
  double lambda;
  physics_t * phys = NULL;

  assert(heq);
  assert(fesymm_oft);
  assert(fesymm_oft->func->mu);

  lees_edw_nhalo(heq->le, &nhalo);
  lees_edw_nlocal(heq->le, nlocal);
  assert(nhalo >= 3);

  physics_ref(&phys);
  physics_lambda(phys, &lambda);

  lees_edw_strides(heq->le, &xs, &ys, &zs);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(heq->le, ic, jc, kc);
	fesymm_oft->func->mu(fesymm_oft, index0 - 2*xs, &mum2);
	fesymm_oft->func->mu(fesymm_oft, index0 - 1*xs, &mum1);
	fesymm_oft->func->mu(fesymm_oft, index0,        &mu00);
	fesymm_oft->func->mu(fesymm_oft, index0 + 1*xs, &mup1);
	fesymm_oft->func->mu(fesymm_oft, index0 + 2*xs, &mup2);

	/* x-direction (between ic-1 and ic) */

	heq->flux->fw[addr_rank0(heq->flux->nsite, index0)]
	  -= 0.25*lambda*(mup1 + mu00 - mum1 - mum2);

	/* ...and between ic and ic+1 */

	heq->flux->fe[addr_rank0(heq->flux->nsite, index0)]
	  -= 0.25*lambda*(mup2 + mup1 - mu00 - mum1);

	/* y direction between jc and jc+1 */

	fesymm_oft->func->mu(fesymm_oft, index0 - 1*ys, &mum1);
	fesymm_oft->func->mu(fesymm_oft, index0 + 1*ys, &mup1);
	fesymm_oft->func->mu(fesymm_oft, index0 + 2*ys, &mup2);

	heq->flux->fy[addr_rank0(heq->flux->nsite, index0)]
	  -= 0.25*lambda*(mup2 + mup1 - mu00 - mum1);

	/* z direction between kc and kc+1 */

	fesymm_oft->func->mu(fesymm_oft, index0 - 1*zs, &mum1);
	fesymm_oft->func->mu(fesymm_oft, index0 + 1*zs, &mup1);
	fesymm_oft->func->mu(fesymm_oft, index0 + 2*zs, &mup2);

	heq->flux->fz[addr_rank0(heq->flux->nsite, index0)]
	  -= 0.25*lambda*(mup2 + mup1 - mu00 - mum1);

	/* Next site */
      }
    }
  }

  return 0;
}


/* TODO: relevance for temperature to be discussed */

/*****************************************************************************
 *
 *  heq_random_flux
 *
 *  This adds (repeat adds) the random contribution to the face
 *  fluxes (advective + diffusive) following Sumesh et al 2011.
 *
 *****************************************************************************/

static int heq_random_flux(heq_t * heq, noise_t * noise) {

  int ic, jc, kc, index0, index1;
  int nsites, nextra;
  int nlocal[3];
  int ia;

  double * rflux;
  double reap[3];
  double kt, lambda, var;
  physics_t * phys = NULL;

  assert(heq);
  assert(heq->le);
  assert(lees_edw_nplane_local(heq->le) == 0);

  /* Variance of the noise from fluctuation dissipation relation */

  physics_ref(&phys);
  physics_kt(phys, &kt);
  physics_lambda(phys, &lambda);
  var = sqrt(2.0*kt*lambda);

  lees_edw_nsites(heq->le, &nsites);
  lees_edw_nlocal(heq->le, nlocal);

  rflux = (double *) malloc(3*nsites*sizeof(double));
  assert(rflux);
  if (rflux == NULL) pe_fatal(heq->pe, "malloc(rflux) failed\n");

  /* We go one site into the halo region to allow all the fluxes to
   * be comupted locally. */

  nextra = 1;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

        index0 = lees_edw_index(heq->le, ic, jc, kc);
        noise_reap_n(noise, index0, 3, reap);

        for (ia = 0; ia < 3; ia++) {
          rflux[addr_rank1(nsites, 3, index0, ia)] = var*reap[ia];
        }

      }
    }
  }

  /* Now accumulate the mid-point fluxes */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(heq->le, ic, jc, kc);

	/* x-direction */

	index1 = lees_edw_index(heq->le, ic-1, jc, kc);
	heq->flux->fw[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, X)] +
		  rflux[addr_rank1(nsites, 3, index1, X)]);

	index1 = lees_edw_index(heq->le, ic+1, jc, kc);
	heq->flux->fe[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, X)] +
		  rflux[addr_rank1(nsites, 3, index1, X)]);

	/* y direction */

	index1 = lees_edw_index(heq->le, ic, jc+1, kc);
	heq->flux->fy[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, Y)] +
		  rflux[addr_rank1(nsites, 3, index1, Y)]);

	/* z direction */

	index1 = lees_edw_index(heq->le, ic, jc, kc+1);
	heq->flux->fz[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, Z)] +
		  rflux[addr_rank1(nsites, 3, index1, Z)]);
      }
    }
  }

  free(rflux);

  return 0;
}


/*****************************************************************************
 *
 *  heq_le_fix_fluxes
 *
 *  Owing to the non-linear calculation for the fluxes,
 *  the LE-interpolated phi field doesn't give a unique
 *  east-west face flux.
 *
 *  This ensures uniqueness, by averaging the relevant
 *  contributions from each side of the plane.
 *
 *  I've retained nf here, as these functions might be useful
 *  for general cases.
 *
 *****************************************************************************/

static int heq_le_fix_fluxes(heq_t * heq, int nf) {

  int nlocal[3];     /* Local system size */
  int mpisz[3];      /* Cartesian size */
  int ip;            /* Index of the plane */
  int ic;            /* Index x location in real system */
  int jc, kc, n;
  int index, index1, index2;
  int nbuffer;

  double ltotal[3];  /* system length */
  double dy;         /* Displacement for current plane */
  double fr;         /* Fractional displacement */
  int jdy;           /* Integral part of displacement */
  int j1, j2;        /* j values in real system to interpolate between */

  double * bufferw;
  double * buffere;

  assert(heq);
  assert(heq->le);

  lees_edw_cartsz(heq->le, mpisz);
  lees_edw_ltot(heq->le, ltotal);

  {
    /* At moment require a copy for device version ... */
    int nplane = lees_edw_nplane_local(heq->le);
    if (nplane) advflux_memcpy(heq->flux, tdpMemcpyDeviceToHost);
  }

  if (mpisz[Y] > 1) {
    /* Parallel */
    heq_le_fix_fluxes_parallel(heq, nf);
  }
  else {
    /* Can do it directly */

    lees_edw_nlocal(heq->le, nlocal);

    nbuffer = nf*nlocal[Y]*nlocal[Z];
    buffere = (double *) malloc(nbuffer*sizeof(double));
    bufferw = (double *) malloc(nbuffer*sizeof(double));
    assert(buffere && bufferw);
    if (buffere == NULL) pe_fatal(heq->pe, "malloc(buffere) failed\n");
    if (bufferw == NULL) pe_fatal(heq->pe, "malloc(bufferw) failed\n");

    for (ip = 0; ip < lees_edw_nplane_local(heq->le); ip++) {

      ic = lees_edw_plane_location(heq->le, ip);

      /* Looking up */

      lees_edw_plane_dy(heq->le, &dy);
      dy = fmod(+dy, ltotal[Y]);
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1; jc <= nlocal[Y]; jc++) {

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    /* This could be replaced by just count++ (check) to addr buffer */
	    index = nf*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    index1 = lees_edw_index(heq->le, ic+1, j1, kc);
	    index2 = lees_edw_index(heq->le, ic+1, j2, kc);

	    bufferw[index] =
	      heq->flux->fw[addr_rank1(heq->flux->nsite, nf, index1, n)]*fr +
	      heq->flux->fw[addr_rank1(heq->flux->nsite, nf, index2, n)]*(1.0-fr);
	  }
	}
      }


      /* Looking down */

      lees_edw_plane_dy(heq->le, &dy);
      dy = fmod(-dy, ltotal[Y]);
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1; jc <= nlocal[Y]; jc++) {

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    index = nf*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    index1 = lees_edw_index(heq->le, ic, j1, kc);
	    index2 = lees_edw_index(heq->le, ic, j2, kc);

	    buffere[index] =
	      heq->flux->fe[addr_rank1(heq->flux->nsite, nf, index1, n)]*fr +
	      heq->flux->fe[addr_rank1(heq->flux->nsite, nf, index2, n)]*(1.0-fr);
	  }
	}
      }

      /* Now average the fluxes. */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    index1 = nf*(nlocal[Z]*(jc-1) + (kc-1)) + n;

	    index = addr_rank1(heq->flux->nsite, nf, lees_edw_index(heq->le,ic,jc,kc), n);
	    heq->flux->fe[index] = 0.5*(heq->flux->fe[index] + bufferw[index1]);
	    index = addr_rank1(heq->flux->nsite, nf, lees_edw_index(heq->le,ic+1,jc,kc), n);
	    heq->flux->fw[index] = 0.5*(heq->flux->fw[index] + buffere[index1]);
	  }
	}
      }

      /* Next plane */
    }

    free(bufferw);
    free(buffere);
  }

  {
    /* At moment require a copy for device version ... */
    int nplane = lees_edw_nplane_local(heq->le);
    if (nplane) advflux_memcpy(heq->flux, tdpMemcpyHostToDevice);
  }

  return 0;
}

/*****************************************************************************
 *
 *  heq_le_fix_fluxes_parallel
 *
 *  Parallel version of the above, where we need to communicate to
 *  get hold of the appropriate fluxes.
 *
 *****************************************************************************/


static int heq_le_fix_fluxes_parallel(heq_t * heq, int nf) {

  int      nhalo;
  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  int ip;                  /* Index of the plane */
  int ic;                  /* Index x location in real system */
  int jc, kc, j1, j2;
  int n, n1, n2;
  int index;
  int ntotal[3];
  double ltotal[3];
  double dy;               /* Displacement for current transforamtion */
  double fre, frw;         /* Fractional displacements */
  int jdy;                 /* Integral part of displacement */

  /* Messages */

  int nsend;               /* N send data */
  int nrecv;               /* N recv data */
  int nrank_s[3];          /* send ranks */
  int nrank_r[3];          /* recv ranks */
  const int tag0 = 1254;
  const int tag1 = 1255;

  double * sbufe = NULL;   /* Send buffer */
  double * sbufw = NULL;   /* Send buffer */
  double * rbufe = NULL;   /* Interpolation buffer */
  double * rbufw = NULL;

  MPI_Comm    le_comm;
  MPI_Request rreq[4], sreq[4];
  MPI_Status  status[4];

  assert(heq);
  assert(heq->le);

  lees_edw_nhalo(heq->le, &nhalo);
  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_ntotal(heq->le, ntotal);
  lees_edw_nlocal_offset(heq->le, noffset);
  lees_edw_ltot(heq->le, ltotal);

  lees_edw_comm(heq->le, &le_comm);

  /* Allocate the temporary buffer */

  nsend = nf*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = nf*(nlocal[Y] + 1)*(nlocal[Z] + 2*nhalo);

  sbufe = (double *) malloc(nsend*sizeof(double));
  sbufw = (double *) malloc(nsend*sizeof(double));

  if (sbufe == NULL) pe_fatal(heq->pe, "malloc(sbufe) failed\n");
  if (sbufw == NULL) pe_fatal(heq->pe, "malloc(sbufw) failed\n");

  rbufe = (double *) malloc(nrecv*sizeof(double));
  rbufw = (double *) malloc(nrecv*sizeof(double));

  if (rbufe == NULL) pe_fatal(heq->pe, "malloc(rbufe) failed\n");
  if (rbufw == NULL) pe_fatal(heq->pe, "malloc(rbufw) failed\n");

  /* One round of communication for each plane */

  for (ip = 0; ip < lees_edw_nplane_local(heq->le); ip++) {

    ic = lees_edw_plane_location(heq->le, ip);

    /* Work out the displacement-dependent quantities */

    lees_edw_plane_dy(heq->le, &dy);
    dy = fmod(+dy, ltotal[Y]);
    jdy = floor(dy);
    frw  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y]. */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];
    assert(j1 > 0);
    assert(j1 <= ntotal[Y]);

    lees_edw_jstart_to_mpi_ranks(heq->le, j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position 1. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 > 0);
    assert(j2 <= nlocal[Y]);

    n1 = nf*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = nf*j2*(nlocal[Z] + 2*nhalo);

    /* Post receives, sends (the wait is later). */

    MPI_Irecv(rbufw,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, rreq);
    MPI_Irecv(rbufw+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm, rreq + 1);

    /* Load send buffer from fw */
    /* (ic+1,j2,1-nhalo) and (ic+1,1,1-nhalo) */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = lees_edw_index(heq->le, ic+1, jc, kc);
	for (n = 0; n < nf; n++) {
	  j1 = nf*(jc - 1)*(nlocal[Z] + 2*nhalo) + nf*(kc + nhalo - 1) + n;
	  assert(j1 >= 0 && j1 < nsend);
	  sbufw[j1] = heq->flux->fw[addr_rank1(heq->flux->nsite, nf, index, n)];
	}
      }
    }

    j1 = (j2 - 1)*nf*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sbufw + j1, n1, MPI_DOUBLE, nrank_s[0], tag0, le_comm, sreq);
    MPI_Issend(sbufw     , n2, MPI_DOUBLE, nrank_s[1], tag1, le_comm, sreq+1);

    /* OTHER WAY */

    lees_edw_plane_dy(heq->le, &dy);
    dy = fmod(-dy, ltotal[Y]);
    jdy = floor(dy);
    fre  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y]. */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];

    lees_edw_jstart_to_mpi_ranks(heq->le, j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = nf*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = nf*j2*(nlocal[Z] + 2*nhalo);

    /* Post new receives, sends, and wait for whole lot to finish. */

    MPI_Irecv(rbufe,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, rreq + 2);
    MPI_Irecv(rbufe+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm, rreq + 3);

    /* Load send buffer from fe */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = lees_edw_index(heq->le, ic, jc, kc);
	for (n = 0; n < nf; n++) {
	  j1 = (jc - 1)*nf*(nlocal[Z] + 2*nhalo) + nf*(kc + nhalo - 1) + n;
	  assert(j1 >= 0 && jc < nsend);
	  sbufe[j1] = heq->flux->fe[addr_rank1(heq->flux->nsite, nf, index, n)];
	}
      }
    }

    j1 = (j2 - 1)*nf*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sbufe + j1, n1, MPI_DOUBLE, nrank_s[0], tag0, le_comm, sreq+2);
    MPI_Issend(sbufe     , n2, MPI_DOUBLE, nrank_s[1], tag1, le_comm, sreq+3);

    MPI_Waitall(4, rreq, status);

    /* Now we've done all the communication, we can update the fluxes
     * using the average of the local value and interpolated buffer
     * value. */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      j1 = (jc - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc - 1 + 1)*(nlocal[Z] + 2*nhalo);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < nf; n++) {
	  index = lees_edw_index(heq->le, ic, jc, kc);
	  heq->flux->fe[addr_rank1(heq->flux->nsite, nf, index, n)]
	    = 0.5*(heq->flux->fe[addr_rank1(heq->flux->nsite, nf, index, n)]
		   + frw*rbufw[nf*(j1 + kc+nhalo-1) + n]
		   + (1.0-frw)*rbufw[nf*(j2 + kc+nhalo-1) + n]);
	  index = lees_edw_index(heq->le, ic+1, jc, kc);
	  heq->flux->fw[addr_rank1(heq->flux->nsite, nf, index, n)]
	    = 0.5*(heq->flux->fw[addr_rank1(heq->flux->nsite, nf, index, n)]
		   + fre*rbufe[nf*(j1 + kc+nhalo-1) + n]
		   + (1.0-fre)*rbufe[nf*(j2 + kc+nhalo-1) + n]);
	}
      }
    }

    /* Clear the sends */
    MPI_Waitall(4, sreq, status);

    /* Next plane */
  }

  free(sbufw);
  free(sbufe);
  free(rbufw);
  free(rbufe);

  return 0;
}

/*****************************************************************************
 *
 *  heq_update_forward_step
 *
 *  Update phi_site at each site in turn via the divergence of the
 *  fluxes. This is an Euler forward step:
 *
 *  phi new = phi old - dt*(flux_out - flux_in)
 *
 *  The time step is the LB time step dt = 1. All sites are processed
 *  to include solid-stored values in the case of Langmuir-Hinshelwood.
 *  It also avoids a conditional on solid/fluid status.
 *
 *****************************************************************************/

static int heq_update_forward_step(heq_t * heq, field_t * temperature) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  double wz = 1.0;

  lees_edw_t * le = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &le);
  lees_edw_strides(heq->le, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  if (nlocal[Z] == 1) wz = 0.0;

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_ufs_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, le, temperature->target, heq->flux->target, ys, wz);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  heq_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void heq_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux,
				  int ys, double wz) {
  int kindex;
  int kiterations;
  int ic, jc, kc, index;
  double temperature;

  assert(ktx);
  assert(le);
  assert(field);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = lees_edw_index(le, ic, jc, kc);
    field_scalar(field, index, &temperature);

    temperature -= (+ flux->fe[addr_rank0(flux->nsite, index)]
	    - flux->fw[addr_rank0(flux->nsite, index)]
	    + flux->fy[addr_rank0(flux->nsite, index)]
	    - flux->fy[addr_rank0(flux->nsite, index - ys)]
	    + wz*flux->fz[addr_rank0(flux->nsite, index)]
	    - wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);

    field_scalar_set(field, index, temperature);
  }

  return;
}

/*****************************************************************************
 *
 *  heq_update_conserve
 *
 *  As above but with the compensated sum at each lattice point.
 *
 *****************************************************************************/

static int heq_update_conserve(heq_t * heq, field_t * temperature) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  double wz = 1.0;

  lees_edw_t * le = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);
  assert(temperature);
  assert(heq->info.conserve);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &le);
  lees_edw_strides(heq->le, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  if (nlocal[Z] == 1) wz = 0.0;

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_csum_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, le, temperature->target, heq->flux->target,
		  heq->csum->target, ys, wz);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}


/****************************************************************************
 *  
 *  correction of phi in order to improve the conservation of phi
 *
 ****************************************************************************/

__global__ void heq_subtract_kernel1(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					temperature_correct_t * correct);
__global__ void heq_subtract_kernel2(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					temperature_correct_t * correct);

static int heq_subtract_sum_temperature_after_forward_step(heq_t * heq, field_t * temperaturef, map_t * map) {

  int nlocal[3];
  dim3 nblk, ntpb;
  
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  temperature_correct_t local = {};
  temperature_correct_t * local_d = NULL;

  assert(heq);
  assert(temperaturef);
  assert(map);

  cs_nlocal(heq->cs, nlocal);
  
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpAssert(tdpMalloc((void **) &local_d, sizeof(temperature_correct_t)));

  /* Work out the local correction... */
  tdpLaunchKernel(heq_subtract_kernel1, nblk, ntpb, 0, 0,
		  ctxt->target, temperaturef->target, map->target, local_d);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  /* Communication stage for global correction... */
  tdpAssert(tdpMemcpy(&local, local_d, sizeof(temperature_correct_t),
		      tdpMemcpyDeviceToHost));

  {
    MPI_Comm comm = MPI_COMM_NULL;
    temperature_correct_t global = {};

    cs_cart_comm(heq->cs, &comm);

    MPI_Allreduce(&local.temperature, &global.temperature, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local.nfluid, &global.nfluid, 1, MPI_INT, MPI_SUM, comm);

    global.T0 = temperaturef->field_init_sum;
    tdpAssert(tdpMemcpy(local_d, &global, sizeof(temperature_correct_t),
			tdpMemcpyHostToDevice));
  }

  /* Apply the correction... */
  tdpLaunchKernel(heq_subtract_kernel2, nblk, ntpb, 0, 0,
		  ctxt->target, temperaturef->target, map->target, local_d);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);
  tdpFree(local_d);

  return 0;
}

/******************************************************************************
 *
 *  heq_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void heq_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				   field_t * field, advflux_t * flux,
				   field_t * csum, int ys, double wz) {
  int kindex;
  int kiterations;
  int ic, jc, kc, index;

  assert(ktx);
  assert(le);
  assert(field);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    kahan_t temperature = {};

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = lees_edw_index(le, ic, jc, kc);

    temperature.sum = field->data[addr_rank1(field->nsites, 1, index, 0)];
    temperature.cs  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

    kahan_add_double(&temperature, -flux->fe[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  flux->fw[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature, -flux->fy[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  flux->fy[addr_rank0(flux->nsite, index - ys)]);
    kahan_add_double(&temperature, -wz*flux->fz[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);

    csum->data[addr_rank1(csum->nsites, 1, index, 0)] = temperature.cs;
    field->data[addr_rank1(field->nsites, 1, index, 0)] = temperature.sum;
  }

  return;
}

/*****************************************************************************
 *
 *
 *  kernel for correction of phi in order to improve the conservation of phi
 *
 *
 ****************************************************************************/

__global__ void heq_subtract_kernel1(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					temperature_correct_t * correct) {
  int kindex;
  int kiterations;
  int index;
  int tid;

  __shared__ double sum_temperature_local[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ int num_fluid_nodes_local[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(map);
  assert(field);
  assert(correct);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  sum_temperature_local[tid] = 0.0;
  num_fluid_nodes_local[tid] = 0;

  correct->temperature = 0.0;
  correct->nfluid = 0;

  for_simt_parallel(kindex, kiterations, 1) {

    int status = 0;
    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    
    index = cs_index(field->cs, ic, jc, kc);
    map_status(map, index, &status);	
    
    if (status == MAP_FLUID) {
      double temperature = 0.0;
      field_scalar(field, index, &temperature);

      sum_temperature_local[tid] += temperature;
      num_fluid_nodes_local[tid] += 1;
    }		  
  }

  /* Reduction */

  __syncthreads();

  if (tid == 0) {
    int nfluid = 0;
    double temperature = 0.0;
    for (int it = 0; it < blockDim.x; it++) {
      nfluid += num_fluid_nodes_local[it];
      temperature    += sum_temperature_local[it];
    }

    tdpAtomicAddInt(&correct->nfluid, nfluid);
    tdpAtomicAddDouble(&correct->temperature, temperature);
  }

  return;
}

/*****************************************************************************
 *
 *  heq_subtract_kernel2
 *
 *****************************************************************************/

__global__ void heq_subtract_kernel2(kernel_ctxt_t * ktx, field_t * field,
					map_t * map, temperature_correct_t * correct) {

  int kindex;
  int kiterations;

  assert(ktx);
  assert(field);
  assert(map);
  assert(correct);
  
  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int status = 0;
    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    int index = cs_index(field->cs, ic, jc, kc);
    double temperature = 0.0;

    field_scalar(field, index, &temperature);
    map_status(map, index, &status);	
    
    if (status == MAP_FLUID) {
      temperature -= (correct->temperature - correct->T0)/correct->nfluid;
    }

    field_scalar_set(field, index, temperature);
  }

  return;
}

/*****************************************************************************
 *
 *  heq_flux_mu_ext
 *
 *  Kernel driver for external chemical potential gradient contribution.
 *
 *****************************************************************************/

static int heq_flux_mu_ext(heq_t * heq) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  heq_kernel_t heq_kernel = {0};

  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &letarget);

  physics_ref(&phys);
  physics_lambda(phys, &heq_kernel.lambda);
  physics_grad_mu(phys,  heq_kernel.gradmu_ex);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_flux_mu_ext_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, heq->flux->target, heq_kernel);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  heq_flux_mu_ext_kernel
 *
 *  Accumulate contributions -m grad mu^ex from external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__global__ void heq_flux_mu_ext_kernel(kernel_ctxt_t * ktx,
					  lees_edw_t * le,
					  advflux_t * flux,
					  heq_kernel_t heq_kernel) {
  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(le);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index0;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = lees_edw_index(le, ic, jc, kc);

    flux->fw[addr_rank0(flux->nsite, index0)] -= heq_kernel.lambda*heq_kernel.gradmu_ex[X];
    flux->fe[addr_rank0(flux->nsite, index0)] -= heq_kernel.lambda*heq_kernel.gradmu_ex[X];
    flux->fy[addr_rank0(flux->nsite, index0)] -= heq_kernel.lambda*heq_kernel.gradmu_ex[Y];
    flux->fz[addr_rank0(flux->nsite, index0)] -= heq_kernel.lambda*heq_kernel.gradmu_ex[Z];
  }

  return;
}
