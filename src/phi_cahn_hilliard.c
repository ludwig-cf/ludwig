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
 *  order parameter mobility. The chemical potential mu is set via
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
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributions:
 *  Thanks to Markus Gross, who helped to validate the noise implementation.
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "physics.h"
#include "advection_s.h"
#include "advection_bcs.h"
#include "util_sum.h"
#include "phi_cahn_hilliard.h"

static int phi_ch_flux_mu1(phi_ch_t * pch, fe_t * fes);
static int phi_ch_update_forward_step(phi_ch_t * pch, field_t * phif);
static int phi_ch_flux_mu_ext(phi_ch_t * pch);

static int phi_ch_update_conserve(phi_ch_t * pch, field_t * phif);

static int phi_ch_le_fix_fluxes(phi_ch_t * pch, int nf);
static int phi_ch_le_fix_fluxes_parallel(phi_ch_t * pch, int nf);
#ifdef NOT_USED
static int phi_ch_flux_mu2(phi_ch_t * pch, fe_t * fes);
static int phi_ch_random_flux(phi_ch_t * pch, noise_t * noise);
#endif
static int phi_ch_subtract_sum_phi_after_forward_step(phi_ch_t * pch,
						      field_t * phif,
						      map_t * map);
/* Utility container */

typedef struct ch_kernel_s ch_kernel_t;
struct ch_kernel_s {
  double mobility;      /* Mobility */
  double gradmu_ex[3];  /* External chemical potential gradient */
};

/* Utility container for corrections to order parameter conservation */

typedef struct phi_correct_s phi_correct_t;
struct phi_correct_s {
  int initial;   /* Are we at t > 0? */
  int nfluid;    /* Fluid volume */
  double phi0;   /* Sum phi at t = 0 */
  double phi;    /* Sum current. */
};


__global__ void phi_ch_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, fe_t * fe,
				       advflux_t * flux, double mobility);
__global__ void phi_ch_flux_mu_ext_kernel(kernel_ctxt_t * ktx,
					  lees_edw_t * le, advflux_t * flux,
					  ch_kernel_t ch);
__global__ void phi_ch_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux,
				  int ys, double wz);
__global__ void phi_ch_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				   field_t * field, advflux_t * flux,
				   field_t * csum, int ys, double wz);

__host__ int phi_ch_dif_flux_driver(phi_ch_t * pch, fe_t * fe,
				    double mobility);
__global__ static void phi_ch_dif_flux_kernel(kernel_ctxt_t * ktx,
					      advflux_t * flux,
					      fe_t * fe,
					      double mobility);

__host__ int phi_ch_var_flux_driver(field_t * var, noise_t * noise,
				    double mobility, double kt);
__global__ static void phi_ch_var_flux_kernel(kernel_ctxt_t * ktx,
					      field_t * var,
					      noise_t * noise,
					      double mktvar);

__host__ int phi_ch_var_flux_acc_driver(phi_ch_t * pch, const field_t * var);
__global__ static void phi_ch_var_flux_acc_kernel(kernel_ctxt_t * ktx,
						  const field_t * var,
						  advflux_t * flux);

/*****************************************************************************
 *
 *  phi_ch_create
 *
 *****************************************************************************/

__host__ int phi_ch_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			   phi_ch_info_t * options,
			   phi_ch_t ** pch) {

  phi_ch_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(le);
  assert(options);
  assert(pch);

  obj = (phi_ch_t *) calloc(1, sizeof(phi_ch_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(phi_ch_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->info = *options;
  advflux_le_create(pe, cs, le, 1, &obj->flux);

  if (obj->info.conserve) {
    field_options_t opts = field_options_ndata_nhalo(1, 0);
    field_create(pe, cs, NULL, "compensated sum", &opts, &obj->csum);
  }

  pe_retain(pe);
  lees_edw_retain(le);

  *pch = obj;

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_free
 *
 *****************************************************************************/

__host__ int phi_ch_free(phi_ch_t * pch) {

  assert(pch);

  lees_edw_free(pch->le);
  pe_free(pch->pe);

  if (pch->csum) field_free(pch->csum);
  advflux_free(pch->flux);
  free(pch);
  
  return 0;
}

/*****************************************************************************
 *
 *  phi_cahn_hilliard
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

int phi_cahn_hilliard(phi_ch_t * pch, fe_t * fe, field_t * phi,
		      hydro_t * hydro, map_t * map,
		      noise_t * noise) {
  int nf;
  int noise_phi = 0;

  assert(pch);
  assert(fe);
  assert(phi);

  if (noise) noise_present(noise, NOISE_PHI, &noise_phi);

  field_nf(phi, &nf);
  assert(nf == 1);

  /* Compute any advective fluxes first, then accumulate diffusive
   * and random fluxes. */

  if (hydro) {
    hydro_u_halo(hydro); /* Reposition to main to prevent repeat */
    hydro_lees_edwards(hydro); /* Repoistion to main ditto */ 
    advection_x(pch->flux, hydro, phi);
  }
  else {
    /* Remember to initialise fluxes to zero for this step */
    advflux_zero(pch->flux);
  }

  if (noise_phi == 0) {
    phi_ch_flux_mu1(pch, fe);
  }
  else {
    /* Bring the mobility and temperature in here, as physics_t is slated
     * for removal; the information is part of the Cahn Hilliard parameters. */
    /* We also have a temporary field for random fluxes computed per site
     * which should really be made permanent if performance is a
     * consideration. */
    physics_t * phys = NULL;
    double mobility = 0.0;
    double kt = 0.0;
    field_options_t opts = field_options_ndata_nhalo(3, 3);
    field_t * var = NULL;

    physics_ref(&phys);
    physics_mobility(phys, &mobility);
    physics_kt(phys, &kt);

    field_create(pch->pe, pch->cs, pch->le, "pch-var", &opts, &var);

    phi_ch_dif_flux_driver(pch, fe, mobility);
    phi_ch_var_flux_driver(var, noise, mobility, kt);
    phi_ch_var_flux_acc_driver(pch, var);

    field_free(var);
  }

  /* External chemical potential gradient (could switch out if zero) */

  phi_ch_flux_mu_ext(pch);

  /* No flux boundaries (diffusive fluxes, and hydrodynamic, if present) */

  if (map) advection_bcs_no_normal_flux(nf, pch->flux, map);

  phi_ch_le_fix_fluxes(pch, nf);

  /* TODO REPLACE 1/2 WITH MEANINGFUL SYMBOLS */
  if (pch->info.conserve == 1) {
    phi_ch_update_conserve(pch, phi);
  }
  else {
    phi_ch_update_forward_step(pch, phi);
    if (pch->info.conserve == 2) {
      phi_ch_subtract_sum_phi_after_forward_step(pch, phi, map);
    }
  }


  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_flux_mu1
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

static int phi_ch_flux_mu1(phi_ch_t * pch, fe_t * fe) {

  int nlocal[3];
  double mobility;
  dim3 nblk, ntpb;
  kernel_info_t limits;

  fe_t * fetarget = NULL;
  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(pch);
  assert(fe);

  lees_edw_nlocal(pch->le, nlocal);
  lees_edw_target(pch->le, &letarget);
  fe->func->target(fe, &fetarget);

  physics_ref(&phys);
  physics_mobility(phys, &mobility);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_ch_flux_mu1_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, fetarget, pch->flux->target, mobility);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_flux_mu1_kernel
 *
 *  Unvectorised kernel.
 *
 *  Accumulate [add to a previously computed advective flux] the
 *  'diffusive' contribution related to the chemical potential. It's
 *  computed everywhere regardless of fluid/solid status.
 *
 *  This is a two point stencil the in the chemical potential,
 *  and the mobility is constant.
 *
 *****************************************************************************/

__global__ void phi_ch_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, fe_t * fe,
				       advflux_t * flux, double mobility) {
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

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);

    index0 = lees_edw_index(le, ic, jc, kc);

    fe->func->mu(fe, index0, &mu0);

    /* x-direction (between ic-1 and ic) */

    index1 = lees_edw_index(le, icm1, jc, kc);
    fe->func->mu(fe, index1, &mu1);
    flux->fw[addr_rank0(flux->nsite, index0)] -= mobility*(mu0 - mu1);

    /* ...and between ic and ic+1 */

    index1 = lees_edw_index(le, icp1, jc, kc);
    fe->func->mu(fe, index1, &mu1);
    flux->fe[addr_rank0(flux->nsite, index0)] -= mobility*(mu1 - mu0);

    /* y direction */

    index1 = lees_edw_index(le, ic, jc+1, kc);
    fe->func->mu(fe, index1, &mu1);
    flux->fy[addr_rank0(flux->nsite, index0)] -= mobility*(mu1 - mu0);

    /* z direction */

    index1 = lees_edw_index(le, ic, jc, kc+1);
    fe->func->mu(fe, index1, &mu1);
    flux->fz[addr_rank0(flux->nsite, index0)] -= mobility*(mu1 - mu0);

    /* Next site */
  }

  return;
}

#ifdef NOT_USED
/*****************************************************************************
 *
 *  phi_ch_flux_mu2
 *
 *  Accumulate [add to previously computed advective fluxes]
 *  diffusive fluxes related to the mobility.
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

static int phi_ch_flux_mu2(phi_ch_t * pch, fe_t * fesymm) {

  int nhalo;
  int nlocal[3];
  int ic, jc, kc;
  int index0;
  int xs, ys, zs;
  double mum2, mum1, mu00, mup1, mup2;
  double mobility;
  physics_t * phys = NULL;

  assert(pch);
  assert(fesymm);
  assert(fesymm->func->mu);

  lees_edw_nhalo(pch->le, &nhalo);
  lees_edw_nlocal(pch->le, nlocal);
  assert(nhalo >= 3);

  physics_ref(&phys);
  physics_mobility(phys, &mobility);

  lees_edw_strides(pch->le, &xs, &ys, &zs);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(pch->le, ic, jc, kc);
	fesymm->func->mu(fesymm, index0 - 2*xs, &mum2);
	fesymm->func->mu(fesymm, index0 - 1*xs, &mum1);
	fesymm->func->mu(fesymm, index0,        &mu00);
	fesymm->func->mu(fesymm, index0 + 1*xs, &mup1);
	fesymm->func->mu(fesymm, index0 + 2*xs, &mup2);

	/* x-direction (between ic-1 and ic) */

	pch->flux->fw[addr_rank0(pch->flux->nsite, index0)]
	  -= 0.25*mobility*(mup1 + mu00 - mum1 - mum2);

	/* ...and between ic and ic+1 */

	pch->flux->fe[addr_rank0(pch->flux->nsite, index0)]
	  -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);

	/* y direction between jc and jc+1 */

	fesymm->func->mu(fesymm, index0 - 1*ys, &mum1);
	fesymm->func->mu(fesymm, index0 + 1*ys, &mup1);
	fesymm->func->mu(fesymm, index0 + 2*ys, &mup2);

	pch->flux->fy[addr_rank0(pch->flux->nsite, index0)]
	  -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);

	/* z direction between kc and kc+1 */

	fesymm->func->mu(fesymm, index0 - 1*zs, &mum1);
	fesymm->func->mu(fesymm, index0 + 1*zs, &mup1);
	fesymm->func->mu(fesymm, index0 + 2*zs, &mup2);

	pch->flux->fz[addr_rank0(pch->flux->nsite, index0)]
	  -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);

	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_random_flux
 *
 *  This adds (repeat adds) the random contribution to the face
 *  fluxes (advective + diffusive) following Sumesh et al 2011.
 *
 *****************************************************************************/

static int phi_ch_random_flux(phi_ch_t * pch, noise_t * noise) {

  int ic, jc, kc, index0, index1;
  int nsites, nextra;
  int nlocal[3];
  int ia;

  double * rflux;
  double reap[3];
  double kt, mobility, var;
  physics_t * phys = NULL;

  assert(pch);
  assert(pch->le);
  assert(lees_edw_nplane_local(pch->le) == 0);

  /* Variance of the noise from fluctuation dissipation relation */

  physics_ref(&phys);
  physics_kt(phys, &kt);
  physics_mobility(phys, &mobility);
  var = sqrt(2.0*kt*mobility);

  lees_edw_nsites(pch->le, &nsites);
  lees_edw_nlocal(pch->le, nlocal);

  rflux = (double *) malloc(3*nsites*sizeof(double));
  assert(rflux);
  if (rflux == NULL) pe_fatal(pch->pe, "malloc(rflux) failed\n");

  /* We go one site into the halo region to allow all the fluxes to
   * be comupted locally. */

  nextra = 1;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

        index0 = lees_edw_index(pch->le, ic, jc, kc);
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

	index0 = lees_edw_index(pch->le, ic, jc, kc);

	/* x-direction */

	index1 = lees_edw_index(pch->le, ic-1, jc, kc);
	pch->flux->fw[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, X)] +
		  rflux[addr_rank1(nsites, 3, index1, X)]);

	index1 = lees_edw_index(pch->le, ic+1, jc, kc);
	pch->flux->fe[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, X)] +
		  rflux[addr_rank1(nsites, 3, index1, X)]);

	/* y direction */

	index1 = lees_edw_index(pch->le, ic, jc+1, kc);
	pch->flux->fy[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, Y)] +
		  rflux[addr_rank1(nsites, 3, index1, Y)]);

	/* z direction */

	index1 = lees_edw_index(pch->le, ic, jc, kc+1);
	pch->flux->fz[addr_rank0(nsites, index0)]
	  += 0.5*(rflux[addr_rank1(nsites, 3, index0, Z)] +
		  rflux[addr_rank1(nsites, 3, index1, Z)]);
      }
    }
  }

  free(rflux);

  return 0;
}
#endif
/*****************************************************************************
 *
 *  phi_ch_le_fix_fluxes
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

static int phi_ch_le_fix_fluxes(phi_ch_t * pch, int nf) {

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

  assert(pch);
  assert(pch->le);

  lees_edw_cartsz(pch->le, mpisz);
  lees_edw_ltot(pch->le, ltotal);

  {
    /* At moment require a copy for device version ... */
    int nplane = lees_edw_nplane_local(pch->le);
    if (nplane) advflux_memcpy(pch->flux, tdpMemcpyDeviceToHost);
  }

  if (mpisz[Y] > 1) {
    /* Parallel */
    phi_ch_le_fix_fluxes_parallel(pch, nf);
  }
  else {
    /* Can do it directly */

    lees_edw_nlocal(pch->le, nlocal);

    nbuffer = nf*nlocal[Y]*nlocal[Z];
    buffere = (double *) malloc(nbuffer*sizeof(double));
    bufferw = (double *) malloc(nbuffer*sizeof(double));
    assert(buffere && bufferw);
    if (buffere == NULL) pe_fatal(pch->pe, "malloc(buffere) failed\n");
    if (bufferw == NULL) pe_fatal(pch->pe, "malloc(bufferw) failed\n");

    for (ip = 0; ip < lees_edw_nplane_local(pch->le); ip++) {

      ic = lees_edw_plane_location(pch->le, ip);

      /* Looking up */

      lees_edw_plane_dy(pch->le, &dy);
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
	    index1 = lees_edw_index(pch->le, ic+1, j1, kc);
	    index2 = lees_edw_index(pch->le, ic+1, j2, kc);

	    bufferw[index] =
	      pch->flux->fw[addr_rank1(pch->flux->nsite, nf, index1, n)]*fr +
	      pch->flux->fw[addr_rank1(pch->flux->nsite, nf, index2, n)]*(1.0-fr);
	  }
	}
      }


      /* Looking down */

      lees_edw_plane_dy(pch->le, &dy);
      dy = fmod(-dy, ltotal[Y]);
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1; jc <= nlocal[Y]; jc++) {

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    index = nf*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    index1 = lees_edw_index(pch->le, ic, j1, kc);
	    index2 = lees_edw_index(pch->le, ic, j2, kc);

	    buffere[index] =
	      pch->flux->fe[addr_rank1(pch->flux->nsite, nf, index1, n)]*fr +
	      pch->flux->fe[addr_rank1(pch->flux->nsite, nf, index2, n)]*(1.0-fr);
	  }
	}
      }

      /* Now average the fluxes. */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    index1 = nf*(nlocal[Z]*(jc-1) + (kc-1)) + n;

	    index = addr_rank1(pch->flux->nsite, nf, lees_edw_index(pch->le,ic,jc,kc), n);
	    pch->flux->fe[index] = 0.5*(pch->flux->fe[index] + bufferw[index1]);
	    index = addr_rank1(pch->flux->nsite, nf, lees_edw_index(pch->le,ic+1,jc,kc), n);
	    pch->flux->fw[index] = 0.5*(pch->flux->fw[index] + buffere[index1]);
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
    int nplane = lees_edw_nplane_local(pch->le);
    if (nplane) advflux_memcpy(pch->flux, tdpMemcpyHostToDevice);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_le_fix_fluxes_parallel
 *
 *  Parallel version of the above, where we need to communicate to
 *  get hold of the appropriate fluxes.
 *
 *****************************************************************************/


static int phi_ch_le_fix_fluxes_parallel(phi_ch_t * pch, int nf) {

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

  assert(pch);
  assert(pch->le);

  lees_edw_nhalo(pch->le, &nhalo);
  lees_edw_nlocal(pch->le, nlocal);
  lees_edw_ntotal(pch->le, ntotal);
  lees_edw_nlocal_offset(pch->le, noffset);
  lees_edw_ltot(pch->le, ltotal);

  lees_edw_comm(pch->le, &le_comm);

  /* Allocate the temporary buffer */

  nsend = nf*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = nf*(nlocal[Y] + 1)*(nlocal[Z] + 2*nhalo);

  sbufe = (double *) malloc(nsend*sizeof(double));
  sbufw = (double *) malloc(nsend*sizeof(double));

  if (sbufe == NULL) pe_fatal(pch->pe, "malloc(sbufe) failed\n");
  if (sbufw == NULL) pe_fatal(pch->pe, "malloc(sbufw) failed\n");

  rbufe = (double *) malloc(nrecv*sizeof(double));
  rbufw = (double *) malloc(nrecv*sizeof(double));

  if (rbufe == NULL) pe_fatal(pch->pe, "malloc(rbufe) failed\n");
  if (rbufw == NULL) pe_fatal(pch->pe, "malloc(rbufw) failed\n");

  /* One round of communication for each plane */

  for (ip = 0; ip < lees_edw_nplane_local(pch->le); ip++) {

    ic = lees_edw_plane_location(pch->le, ip);

    /* Work out the displacement-dependent quantities */

    lees_edw_plane_dy(pch->le, &dy);
    dy = fmod(+dy, ltotal[Y]);
    jdy = floor(dy);
    frw  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y]. */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];
    assert(j1 > 0);
    assert(j1 <= ntotal[Y]);

    lees_edw_jstart_to_mpi_ranks(pch->le, j1, nrank_s, nrank_r);

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
	index = lees_edw_index(pch->le, ic+1, jc, kc);
	for (n = 0; n < nf; n++) {
	  j1 = nf*(jc - 1)*(nlocal[Z] + 2*nhalo) + nf*(kc + nhalo - 1) + n;
	  assert(j1 >= 0 && j1 < nsend);
	  sbufw[j1] = pch->flux->fw[addr_rank1(pch->flux->nsite, nf, index, n)];
	}
      }
    }

    j1 = (j2 - 1)*nf*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sbufw + j1, n1, MPI_DOUBLE, nrank_s[0], tag0, le_comm, sreq);
    MPI_Issend(sbufw     , n2, MPI_DOUBLE, nrank_s[1], tag1, le_comm, sreq+1);

    /* OTHER WAY */

    lees_edw_plane_dy(pch->le, &dy);
    dy = fmod(-dy, ltotal[Y]);
    jdy = floor(dy);
    fre  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= ntotal[Y]. */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*ntotal[Y]) % ntotal[Y];

    lees_edw_jstart_to_mpi_ranks(pch->le, j1, nrank_s, nrank_r);

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
	index = lees_edw_index(pch->le, ic, jc, kc);
	for (n = 0; n < nf; n++) {
	  j1 = (jc - 1)*nf*(nlocal[Z] + 2*nhalo) + nf*(kc + nhalo - 1) + n;
	  assert(j1 >= 0 && jc < nsend);
	  sbufe[j1] = pch->flux->fe[addr_rank1(pch->flux->nsite, nf, index, n)];
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
	  index = lees_edw_index(pch->le, ic, jc, kc);
	  pch->flux->fe[addr_rank1(pch->flux->nsite, nf, index, n)]
	    = 0.5*(pch->flux->fe[addr_rank1(pch->flux->nsite, nf, index, n)]
		   + frw*rbufw[nf*(j1 + kc+nhalo-1) + n]
		   + (1.0-frw)*rbufw[nf*(j2 + kc+nhalo-1) + n]);
	  index = lees_edw_index(pch->le, ic+1, jc, kc);
	  pch->flux->fw[addr_rank1(pch->flux->nsite, nf, index, n)]
	    = 0.5*(pch->flux->fw[addr_rank1(pch->flux->nsite, nf, index, n)]
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
 *  phi_ch_update_forward_step
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

static int phi_ch_update_forward_step(phi_ch_t * pch, field_t * phif) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  double wz = 1.0;

  lees_edw_t * le = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  lees_edw_nlocal(pch->le, nlocal);
  lees_edw_target(pch->le, &le);
  lees_edw_strides(pch->le, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  if (nlocal[Z] == 1) wz = 0.0;

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_ch_ufs_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, le, phif->target, pch->flux->target, ys, wz);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  phi_ch_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void phi_ch_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux,
				  int ys, double wz) {
  int kindex;
  int kiterations;
  int ic, jc, kc, index;
  double phi;

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
    field_scalar(field, index, &phi);

    phi -= (+ flux->fe[addr_rank0(flux->nsite, index)]
	    - flux->fw[addr_rank0(flux->nsite, index)]
	    + flux->fy[addr_rank0(flux->nsite, index)]
	    - flux->fy[addr_rank0(flux->nsite, index - ys)]
	    + wz*flux->fz[addr_rank0(flux->nsite, index)]
	    - wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);

    field_scalar_set(field, index, phi);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_update_conserve
 *
 *  As above but with the compensated sum at each lattice point.
 *
 *****************************************************************************/

static int phi_ch_update_conserve(phi_ch_t * pch, field_t * phi) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  double wz = 1.0;

  lees_edw_t * le = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(pch);
  assert(phi);
  assert(pch->info.conserve);

  lees_edw_nlocal(pch->le, nlocal);
  lees_edw_target(pch->le, &le);
  lees_edw_strides(pch->le, &xs, &ys, &zs);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  if (nlocal[Z] == 1) wz = 0.0;

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_ch_csum_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, le, phi->target, pch->flux->target,
		  pch->csum->target, ys, wz);

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

__global__ void phi_ch_subtract_kernel1(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					phi_correct_t * correct);
__global__ void phi_ch_subtract_kernel2(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					phi_correct_t * correct);

static int phi_ch_subtract_sum_phi_after_forward_step(phi_ch_t * pch, field_t * phif, map_t * map) {

  int nlocal[3];
  dim3 nblk, ntpb;
  
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  phi_correct_t local = {0};
  phi_correct_t * local_d = NULL;

  assert(pch);
  assert(phif);
  assert(map);

  cs_nlocal(pch->cs, nlocal);
  
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpAssert(tdpMalloc((void **) &local_d, sizeof(phi_correct_t)));

  /* Work out the local correction... */
  tdpLaunchKernel(phi_ch_subtract_kernel1, nblk, ntpb, 0, 0,
		  ctxt->target, phif->target, map->target, local_d);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  /* Communication stage for global correction... */
  tdpAssert(tdpMemcpy(&local, local_d, sizeof(phi_correct_t),
		      tdpMemcpyDeviceToHost));

  {
    MPI_Comm comm = MPI_COMM_NULL;
    phi_correct_t global = {0};

    cs_cart_comm(pch->cs, &comm);

    MPI_Allreduce(&local.phi, &global.phi, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local.nfluid, &global.nfluid, 1, MPI_INT, MPI_SUM, comm);

    global.phi0 = phif->field_init_sum;
    tdpAssert(tdpMemcpy(local_d, &global, sizeof(phi_correct_t),
			tdpMemcpyHostToDevice));
  }

  /* Apply the correction... */
  tdpLaunchKernel(phi_ch_subtract_kernel2, nblk, ntpb, 0, 0,
		  ctxt->target, phif->target, map->target, local_d);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);
  tdpFree(local_d);

  return 0;
}

/******************************************************************************
 *
 *  phi_ch_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void phi_ch_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
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

    kahan_t phi = {0};

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = lees_edw_index(le, ic, jc, kc);

    phi.sum = field->data[addr_rank1(field->nsites, 1, index, 0)];
    phi.cs  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

    kahan_add_double(&phi, -flux->fe[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&phi,  flux->fw[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&phi, -flux->fy[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&phi,  flux->fy[addr_rank0(flux->nsite, index - ys)]);
    kahan_add_double(&phi, -wz*flux->fz[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&phi,  wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);

    csum->data[addr_rank1(csum->nsites, 1, index, 0)] = phi.cs;
    field->data[addr_rank1(field->nsites, 1, index, 0)] = phi.sum;
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

__global__ void phi_ch_subtract_kernel1(kernel_ctxt_t * ktx,
					field_t * field, map_t * map,
					phi_correct_t * correct) {
  int kindex;
  int kiterations;
  int index;
  int tid;

  __shared__ double sum_phi_local[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ int num_fluid_nodes_local[TARGET_MAX_THREADS_PER_BLOCK];

  assert(ktx);
  assert(map);
  assert(field);
  assert(correct);

  kiterations = kernel_iterations(ktx);

  tid = threadIdx.x;
  sum_phi_local[tid] = 0.0;
  num_fluid_nodes_local[tid] = 0;

  correct->phi = 0.0;
  correct->nfluid = 0;

  for_simt_parallel(kindex, kiterations, 1) {

    int status = 0;
    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);
    
    index = cs_index(field->cs, ic, jc, kc);
    map_status(map, index, &status);	
    
    if (status == MAP_FLUID) {
      double phi = 0.0;
      field_scalar(field, index, &phi);

      sum_phi_local[tid] += phi;
      num_fluid_nodes_local[tid] += 1;
    }		  
  }

  /* Reduction */

  __syncthreads();

  if (tid == 0) {
    int nfluid = 0;
    double phi = 0.0;
    for (int it = 0; it < blockDim.x; it++) {
      nfluid += num_fluid_nodes_local[it];
      phi    += sum_phi_local[it];
    }

    tdpAtomicAddInt(&correct->nfluid, nfluid);
    tdpAtomicAddDouble(&correct->phi, phi);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_subtract_kernel2
 *
 *****************************************************************************/

__global__ void phi_ch_subtract_kernel2(kernel_ctxt_t * ktx, field_t * field,
					map_t * map, phi_correct_t * correct) {

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
    double phi = 0.0;

    field_scalar(field, index, &phi);
    map_status(map, index, &status);	
    
    if (status == MAP_FLUID) {
      phi -= (correct->phi - correct->phi0)/correct->nfluid;
    }

    field_scalar_set(field, index, phi);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_flux_mu_ext
 *
 *  Kernel driver for external chemical potential gradient contribution.
 *
 *****************************************************************************/

static int phi_ch_flux_mu_ext(phi_ch_t * pch) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  ch_kernel_t ch = {0};

  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(pch);

  lees_edw_nlocal(pch->le, nlocal);
  lees_edw_target(pch->le, &letarget);

  physics_ref(&phys);
  physics_mobility(phys, &ch.mobility);
  physics_grad_mu(phys,  ch.gradmu_ex);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(pch->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(phi_ch_flux_mu_ext_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, pch->flux->target, ch);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_flux_mu_ext_kernel
 *
 *  Accumulate contributions -m grad mu^ex from external chemical
 *  potential gradient.
 *
 *****************************************************************************/

__global__ void phi_ch_flux_mu_ext_kernel(kernel_ctxt_t * ktx,
					  lees_edw_t * le,
					  advflux_t * flux,
					  ch_kernel_t ch) {
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

    flux->fw[addr_rank0(flux->nsite, index0)] -= ch.mobility*ch.gradmu_ex[X];
    flux->fe[addr_rank0(flux->nsite, index0)] -= ch.mobility*ch.gradmu_ex[X];
    flux->fy[addr_rank0(flux->nsite, index0)] -= ch.mobility*ch.gradmu_ex[Y];
    flux->fz[addr_rank0(flux->nsite, index0)] -= ch.mobility*ch.gradmu_ex[Z];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_dif_flux_driver
 *
 *  Driver for fourth order diffusive fluxes.
 *
 *****************************************************************************/

__host__ int phi_ch_dif_flux_driver(phi_ch_t * pch, fe_t * fe,
				    double mobility) {
  int nlocal[3] = {0};
  fe_t * fetarget = NULL;

  assert(pch);
  assert(fe);

  cs_nlocal(pch->cs, nlocal);
  fe->func->target(fe, &fetarget);

  {
    dim3 nblk, ntpb;
    kernel_info_t lim = {1, nlocal[X], 0, nlocal[Y], 0, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;

    kernel_ctxt_create(pch->cs, 1, lim, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(phi_ch_dif_flux_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, pch->flux->target, fetarget, mobility);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_dif_flux_kernel
 *
 *  Accumulate [add to previously computed advective fluxes]
 *  diffusive fluxes related to the mobility.
 *
 *  This version is based on Sumesh et al to allow correct
 *  treatment of noise. The fluxes are calculated via
 *
 *  grad_x mu = 0.5*(mu(i+1) - mu(i-1)) etc
 *  flux_x(x + 1/2) = -0.5*M*(grad_x mu(i) + grad_x mu(i+1)) etc
 *
 *  In contrast to Sumesh et al., we don't have 'diagonal' fluxes.
 *  There are no Lees Edwards planes implemented, but we maintain
 *  the distiction between east and west in the fluxes.
 *
 *****************************************************************************/

__global__ static void phi_ch_dif_flux_kernel(kernel_ctxt_t * ktx,
					      advflux_t * flux,
					      fe_t * fe,
					      double mobility) {
  int kindex = 0;
  int kiterations = 0;

  assert(ktx);
  assert(flux);
  assert(fe);
  assert(fe->func->mu);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int indexm2 = kernel_coords_index(ktx, ic-2, jc, kc);
    int indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
    int index00 = kernel_coords_index(ktx, ic,   jc, kc);
    int indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);
    int indexp2 = kernel_coords_index(ktx, ic+2, jc, kc);

    double mum2 = 0.0;
    double mum1 = 0.0;
    double mu00 = 0.0;
    double mup1 = 0.0;
    double mup2 = 0.0;

    fe->func->mu(fe, indexm2, &mum2);
    fe->func->mu(fe, indexm1, &mum1);
    fe->func->mu(fe, index00, &mu00);
    fe->func->mu(fe, indexp1, &mup1);
    fe->func->mu(fe, indexp2, &mup2);

    /* x-direction (between ic-1 and ic) */

    flux->fw[addr_rank0(flux->nsite, index00)]
      -= 0.25*mobility*(mup1 + mu00 - mum1 - mum2);

    /* ...and between ic and ic+1 */

    flux->fe[addr_rank0(flux->nsite, index00)]
      -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);

    /* y direction between jc and jc+1 */

    indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
    indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);
    indexp2 = kernel_coords_index(ktx, ic, jc+2, kc);

    fe->func->mu(fe, indexm1, &mum1);
    fe->func->mu(fe, indexp1, &mup1);
    fe->func->mu(fe, indexp2, &mup2);

    flux->fy[addr_rank0(flux->nsite, index00)]
      -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);

    /* z direction between kc and kc+1 */

    indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
    indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);
    indexp2 = kernel_coords_index(ktx, ic, jc, kc+2);

    fe->func->mu(fe, indexm1, &mum1);
    fe->func->mu(fe, indexp1, &mup1);
    fe->func->mu(fe, indexp2, &mup2);

    flux->fz[addr_rank0(flux->nsite, index00)]
      -= 0.25*mobility*(mup2 + mup1 - mu00 - mum1);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_var_flux_driver
 *
 *  Compute fluctuating contribution to fluxes per site. The results must
 *  be translated to face-fluxes before appearing in the final update.
 *
 *****************************************************************************/

__host__ int phi_ch_var_flux_driver(field_t * var,
				    noise_t * noise,
				    double mobility,
				    double kt) {
  int nlocal[3] = {0};

  cs_nlocal(var->cs, nlocal);

  {
    /* Fluctuation dissipation says ... */
    double mktvar = sqrt(2.0*mobility*kt);
    /* Limits have nextra = 1 site at each end */
    kernel_info_t lim = {0, nlocal[X]+1, 0, nlocal[Y]+1, 0, nlocal[Z]+1};
    kernel_ctxt_t * ctxt = NULL;
    dim3 nblk, ntpb;

    kernel_ctxt_create(var->cs, 1, lim, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(phi_ch_var_flux_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, var->target, noise->target, mktvar);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_var_flux_kernel
 *
 *  Compute per site.
 *  Variance of the noise is from fluctuation dissipation relation.
 *
 *****************************************************************************/

__global__ static void phi_ch_var_flux_kernel(kernel_ctxt_t * ktx,
					      field_t * var,
					      noise_t * noise,
					      double mktvar) {
  int kindex = 0;
  int kiterations = 0;

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index0 = kernel_coords_index(ktx, ic, jc, kc);

    double reap[3] = {0};
    noise_reap_n(noise, index0, 3, reap);

    var->data[addr_rank1(var->nsites, 3, index0, X)] = mktvar*reap[X];
    var->data[addr_rank1(var->nsites, 3, index0, Y)] = mktvar*reap[Y];
    var->data[addr_rank1(var->nsites, 3, index0, Z)] = mktvar*reap[Z];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_var_flux_acc_driver
 *
 *  Accumulate the random fluxes computed at sites to the relevant face
 *  faces. This is just a simple average of the two site either side of
 *  a given cell face.
 *
 *****************************************************************************/

__host__ int phi_ch_var_flux_acc_driver(phi_ch_t * pch, const field_t * var) {

  int nlocal[3] = {0};

  cs_nlocal(pch->cs, nlocal);

  {
    kernel_info_t lim = {1, nlocal[X], 0, nlocal[Y], 0, nlocal[Z]};
    kernel_ctxt_t * ctxt = NULL;
    dim3 nblk, ntpb;

    kernel_ctxt_create(pch->cs, 1, lim, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(phi_ch_var_flux_acc_kernel, nblk, ntpb, 0, 0,
		    ctxt->target, var->target, pch->flux->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_ch_var_flux_acc_kernel
 *
 *  Accumulate fluctuation-dissipation part of face fluxes.
 *
 *****************************************************************************/

__global__ static void phi_ch_var_flux_acc_kernel(kernel_ctxt_t * ktx,
						  const field_t * var,
						  advflux_t * flux) {
  int kindex = 0;
  int kiterations = 0;

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic = kernel_coords_ic(ktx, kindex);
    int jc = kernel_coords_jc(ktx, kindex);
    int kc = kernel_coords_kc(ktx, kindex);

    int index0 = kernel_coords_index(ktx, ic, jc, kc);

    /* x-direction (west face) */
    {
      int index1 = kernel_coords_index(ktx, ic-1, jc, kc);
      int vaddr0 = addr_rank1(var->nsites, 3, index0, X);
      int vaddr1 = addr_rank1(var->nsites, 3, index1, X);
      flux->fw[addr_rank0(flux->nsite, index0)]
	+= 0.5*(var->data[vaddr0] + var->data[vaddr1]);
    }

    /* x-direction (east face) */
    {
      int index1 = kernel_coords_index(ktx, ic+1, jc, kc);
      int vaddr0 = addr_rank1(var->nsites, 3, index0, X);
      int vaddr1 = addr_rank1(var->nsites, 3, index1, X);
      flux->fe[addr_rank0(flux->nsite, index0)]
	+= 0.5*(var->data[vaddr0] + var->data[vaddr1]);
    }

    /* y direction */
    {
      int index1 = kernel_coords_index(ktx, ic, jc+1, kc);
      int vaddr0 = addr_rank1(var->nsites, 3, index0, Y);
      int vaddr1 = addr_rank1(var->nsites, 3, index1, Y);
      flux->fy[addr_rank0(flux->nsite, index0)]
	  += 0.5*(var->data[vaddr0] + var->data[vaddr1]);
    }

    /* z direction */
    {
      int index1 = kernel_coords_index(ktx, ic, jc, kc+1);
      int vaddr0 = addr_rank1(var->nsites, 3, index0, Z);
      int vaddr1 = addr_rank1(var->nsites, 3, index1, Z);
      flux->fz[addr_rank0(flux->nsite, index0)]
	+= 0.5*(var->data[vaddr0] + var->data[vaddr1]);
    }
  }

  return;
}
