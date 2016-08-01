/*****************************************************************************
 *
 *  phi_force.c
 *
 *  Computes the force on the fluid from the thermodynamic sector
 *  via the divergence of the chemical stress. Its calculation as
 *  a divergence ensures momentum is conserved.
 *
 *  Note that the stress may be asymmetric.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "wall.h"
#include "field_grad_s.h"
#include "field_s.h"
#include "hydro_s.h"
#include "pth_s.h"
#include "timer.h"
#include "phi_force.h"

__global__
void phi_force_fluid_kernel_v(kernel_ctxt_t * ktx, pth_t * pth,
			      hydro_t * hydro);

static int phi_force_calculation_fluid(pth_t * pth, fe_t * fe, field_t * q, 
				       field_grad_t * qgrad, hydro_t * hydro);

static int phi_force_compute_fluxes(pth_t * pth, fe_t * fe, double * fxe,
				    double * fxw,
				    double * fxy,
				    double * fxz);
static int phi_force_flux_divergence(pth_t * pth, hydro_t * hydro, double * fe,
				     double * fw, double * fy, double * fz);
static int phi_force_flux_fix_local(pth_t * pth, double * fluxe, double * fluxw);
static int phi_force_flux_divergence_with_fix(pth_t * pth,
					      hydro_t * hydro, double * fe,
					      double * fw,
					      double * fy, double * fz);
static int phi_force_flux(pth_t * pth, fe_t * fe, hydro_t * hydro);
static int phi_force_wallx(pth_t * pth, fe_t * fe, double * fxe, double * fxw);
static int phi_force_wally(pth_t * pth, fe_t * fe, double * fy);
static int phi_force_wallz(pth_t * pth, fe_t * fe, double * fz);

static int phi_force_fluid_phi_gradmu(pth_t * pth, fe_t * fe, field_t * phi,
				      hydro_t * hydro);

/*****************************************************************************
 *
 *  phi_force_calculation
 *
 *  Driver routine to compute the body force on fluid from phi sector.
 *
 *  If hydro is NULL, we assume hydroynamics is not present, so there
 *  is no force.
 *
 *****************************************************************************/

__host__ int phi_force_calculation(pth_t * pth, fe_t * fe,
				   field_t * phi, field_t* q,
				   field_grad_t* q_grad, hydro_t * hydro) {

  if (pth == NULL) return 0;
  if (pth->method == PTH_METHOD_NO_FORCE) return 0;
  if (hydro == NULL) return 0; 

  if (le_get_nplane_total() > 0 || wall_present()) {
    /* Must use the flux method for LE planes */
    /* Also convenient for plane walls */

    phi_force_flux(pth, fe, hydro);
  }
  else {
    switch (pth->method) {
    case PTH_METHOD_DIVERGENCE:
      phi_force_calculation_fluid(pth, fe, q, q_grad, hydro);
      break;
    case PTH_METHOD_GRADMU:
      phi_force_fluid_phi_gradmu(pth, fe, phi, hydro);
      break;
    default:
      assert(0);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *
 *    F_alpha = nalba_beta Pth_alphabeta
 *
 *  using a simple six-point stencil.
 *
 *  Kernel driver.
 *
 *****************************************************************************/

__host__ int phi_force_calculation_fluid(pth_t * pth, fe_t * fe,
					 field_t * q,
					 field_grad_t * q_grad,
					 hydro_t * hydro) {
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  
  assert(pth);
  assert(fe);
  assert(hydro);

  pth_stress_compute(pth, fe);

  coords_nlocal(nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  TIMER_start(TIMER_PHI_FORCE_CALC);

  __host_launch(phi_force_fluid_kernel_v, nblk, ntpb, ctxt->target,
		pth->target, hydro->target);
  targetSynchronize();

  TIMER_stop(TIMER_PHI_FORCE_CALC);

  kernel_ctxt_free(ctxt);

  return 0;
}


/*****************************************************************************
 *
 *  phi_force_calculation
 *
 *  Increment force at each lattice site.
 *
 *****************************************************************************/

__global__
void phi_force_fluid_kernel_v(kernel_ctxt_t * ktx, pth_t * pth,
			      hydro_t * hydro) {

  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(pth);
  assert(hydro);

  kiterations = kernel_vector_iterations(ktx);

  __target_simt_parallel_for(kindex, kiterations, NSIMDVL) {

    int iv;
    int ia, ib;
    int index;                   /* first index in vector block */
    int nsites;
    int ic[NSIMDVL];             /* ic for this iteration */
    int jc[NSIMDVL];             /* jc for this iteration */
    int kc[NSIMDVL];             /* kc ditto */
    int pm[NSIMDVL];             /* ordinate +/- 1 */
    int maskv[NSIMDVL];          /* = 0 if not kernel site, 1 otherwise */
    int index1[NSIMDVL];
    double pth0[3][3][NSIMDVL];
    double pth1[3][3][NSIMDVL];
    double force[3][NSIMDVL];


    index = kernel_baseindex(ktx, kindex);
    kernel_coords_v(ktx, kindex, ic, jc, kc);

    kernel_mask_v(ktx, ic, jc, kc, maskv);

    nsites = pth->nsites;

    /* Compute pth at current point */
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) {
	  pth0[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index+iv,ia,ib)];
	}
      }
    }

    /* Compute differences */

    __targetILP__(iv) pm[iv] = ic[iv] + maskv[iv];
    kernel_coords_index_v(ktx, pm, jc, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] = -0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
    }


    __targetILP__(iv) pm[iv] = ic[iv] - maskv[iv];
    kernel_coords_index_v(ktx, pm, jc, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] += 0.5*(pth1[ia][X][iv] + pth0[ia][X][iv]);
    }

    __targetILP__(iv) pm[iv] = jc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, pm, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] -= 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
    }

    __targetILP__(iv) pm[iv] = jc[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ic, pm, kc, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) {
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] += 0.5*(pth1[ia][Y][iv] + pth0[ia][Y][iv]);
    }

    __targetILP__(iv) pm[iv] = kc[iv] + maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, pm, index1);

    for (ia = 0; ia < 3; ia++){
      for (ib = 0; ib < 3; ib++){
	__targetILP__(iv) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] -= 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
    }

    __targetILP__(iv) pm[iv] = kc[iv] - maskv[iv];
    kernel_coords_index_v(ktx, ic, jc, pm, index1);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) { 
	  pth1[ia][ib][iv] = pth->str[addr_rank2(nsites,3,3,index1[iv],ia,ib)];
	}
      }
    }
    
    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) force[ia][iv] += 0.5*(pth1[ia][Z][iv] + pth0[ia][Z][iv]);
    }

    /* Store the force on lattice */

    for (ia = 0; ia < 3; ia++) { 
      __targetILP__(iv) { 
	hydro->f[addr_rank1(hydro->nsite,NHDIM,index+iv,ia)]
	  += force[ia][iv]*maskv[iv];
      }
    }
    /* Next site */
  }
  
  return;
}


/*****************************************************************************
 *
 *  phi_force_fluid_phi_gradmu
 *
 *  This computes and stores the force on the fluid via
 *    f_a = - phi \nabla_a mu
 *
 *  which is appropriate for the symmtric and Brazovskii
 *  free energies, It is provided as a choice.
 *
 *  The gradient of the chemical potential is computed as
 *    grad_x mu = 0.5*(mu(i+1) - mu(i-1)) etc
 *  Lees-Edwards planes are allowed for.
 *
 *****************************************************************************/

static int phi_force_fluid_phi_gradmu(pth_t * pth, fe_t * fe, field_t * fphi,
				      hydro_t * hydro) {

  int ic, jc, kc, icm1, icp1;
  int index0, indexm1, indexp1;
  int nhalo;
  int nlocal[3];
  int zs, ys;
  double phi, mum1, mup1;
  double force[3];

  assert(fphi);
  assert(hydro);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  assert(nhalo >= 2);

  /* Memory strides */
  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);
	field_scalar(fphi, index0, &phi);

        indexm1 = le_site_index(icm1, jc, kc);
        indexp1 = le_site_index(icp1, jc, kc);

	fe->func->mu(fe, indexm1, &mum1);
	fe->func->mu(fe, indexp1, &mup1);

        force[X] = -phi*0.5*(mup1 - mum1);

	fe->func->mu(fe, index0 - ys, &mum1);
	fe->func->mu(fe, index0 + ys, &mup1);

        force[Y] = -phi*0.5*(mup1 - mum1);

	fe->func->mu(fe, index0 - zs, &mum1);
	fe->func->mu(fe, index0 + zs, &mup1);

        force[Z] = -phi*0.5*(mup1 - mum1);

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index0, force);

	/* Next site */
      }
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  phi_force_flux
 *
 *  Here we compute the momentum fluxes, the divergence of which will
 *  give rise to the force on the fluid.
 *
 *  The flux form is used to ensure conservation, and to allow
 *  the appropriate corrections when LE planes are present.
 *
 *****************************************************************************/

static int phi_force_flux(pth_t * pth, fe_t * fe, hydro_t * hydro) {

  int n;
  int fix_fluxes = 1;

  double * fluxe;
  double * fluxw;
  double * fluxy;
  double * fluxz;

  assert(hydro);

  n = coords_nsites();

  fluxe = (double *) malloc(3*n*sizeof(double));
  fluxw = (double *) malloc(3*n*sizeof(double));
  fluxy = (double *) malloc(3*n*sizeof(double));
  fluxz = (double *) malloc(3*n*sizeof(double));

  if (fluxe == NULL) fatal("malloc(fluxe) force failed");
  if (fluxw == NULL) fatal("malloc(fluxw) force failed");
  if (fluxy == NULL) fatal("malloc(fluxy) force failed");
  if (fluxz == NULL) fatal("malloc(fluxz) force failed");

  phi_force_compute_fluxes(pth, fe, fluxe, fluxw, fluxy, fluxz);

  if (wall_at_edge(X)) phi_force_wallx(pth, fe, fluxe, fluxw);
  if (wall_at_edge(Y)) phi_force_wally(pth, fe, fluxy);
  if (wall_at_edge(Z)) phi_force_wallz(pth, fe, fluxz);

  if (fix_fluxes || wall_present()) {
    phi_force_flux_fix_local(pth, fluxe, fluxw);
    phi_force_flux_divergence(pth, hydro, fluxe, fluxw, fluxy, fluxz);
  }
  else {
    phi_force_flux_divergence_with_fix(pth, hydro, fluxe, fluxw, fluxy, fluxz);
  }

  free(fluxz);
  free(fluxy);
  free(fluxw);
  free(fluxe);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_compute_fluxes
 *
 *  Linearly interpolate the chemical stress to the cell faces to get
 *  the momentum fluxes.
 *
 *  This is designed for LE planes; the chemical stress routine must
 *  be called directly, as phi_force_stress cannot handle the planes.
 *
 *****************************************************************************/


static int phi_force_compute_fluxes(pth_t * pth, fe_t * fe,
				    double * fluxe, double * fluxw,
				    double * fluxy, double * fluxz) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];

  int nsites;

  coords_nlocal(nlocal);
  nsites = coords_nsites();

  assert(coords_nhalo() >= 2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	/* Compute pth at current point */
	fe->func->stress(fe, index, pth0);

	/* fluxw_a = (1/2)[P(i, j, k) + P(i-1, j, k)]_xa */
	
	index1 = le_site_index(icm1, jc, kc);

	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[addr_rank1(nsites,3,index,ia)] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxe_a = (1/2)[P(i, j, k) + P(i+1, j, k)_xa */

	index1 = le_site_index(icp1, jc, kc);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nsites,3,index,ia)] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxy_a = (1/2)[P(i, j, k) + P(i, j+1, k)]_ya */

	index1 = le_site_index(ic, jc+1, kc);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[addr_rank1(nsites,3,index,ia)] = 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	/* fluxz_a = (1/2)[P(i, j, k) + P(i, j, k+1)]_za */

	index1 = le_site_index(ic, jc, kc+1);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[addr_rank1(nsites,3,index,ia)] = 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_flux_divergence
 *
 *  Take the diverence of the momentum fluxes to get a force on the
 *  fluid site.
 *
 *****************************************************************************/

static int phi_force_flux_divergence(pth_t * pth, hydro_t * hydro,
				     double * fluxe, double * fluxw,
				     double * fluxy, double * fluxz) {
  int nlocal[3];
  int ic, jc, kc, ia;
  int index, indexj, indexk;

  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  assert(hydro);
  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = le_site_index(ic, jc, kc);

	indexj = le_site_index(ic, jc-1, kc);
	indexk = le_site_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  hydro->f[addr_rank1(le_nsites(), NHDIM,index,ia)]
	    += -(+ fluxe[addr_rank1(nSites,3,index,ia)]
		 - fluxw[addr_rank1(nSites,3,index,ia)]
		 + fluxy[addr_rank1(nSites,3,index,ia)]
		 - fluxy[addr_rank1(nSites,3,indexj,ia)]
		 + fluxz[addr_rank1(nSites,3,index,ia)]
		 - fluxz[addr_rank1(nSites,3,indexk,ia)]);

	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_flux_divergence_with_fix
 *
 *  Take the diverence of the momentum fluxes to get a force on the
 *  fluid site.
 *
 *  It is intended that these fluxes are uncorrected, and that a
 *  global constraint on the total force is enforced. This costs
 *  one Allreduce in pe_comm() per call. 
 *
 *****************************************************************************/

static int phi_force_flux_divergence_with_fix(pth_t * pth,
					      hydro_t * hydro,
					      double * fluxe, double * fluxw,
					      double * fluxy,
					      double * fluxz) {
  int nlocal[3];
  int ic, jc, kc, index, ia;
  int indexj, indexk;
  double f[3];

  double fsum_local[3];
  double fsum[3];
  double rv;

  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  assert(hydro);
  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  assert(0); /* SHIT NO TEST? */

  coords_nlocal(nlocal);

  for (ia = 0; ia < 3; ia++) {
    fsum_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = le_site_index(ic, jc, kc);
	indexj = le_site_index(ic, jc-1, kc);
	indexk = le_site_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia]
	    = - (+ fluxe[addr_rank1(nSites,3,index,ia)]
		 - fluxw[addr_rank1(nSites,3,index,ia)]
		 + fluxy[addr_rank1(nSites,3,index,ia)]
		 - fluxy[addr_rank1(nSites,3,indexj,ia)]
		 + fluxz[addr_rank1(nSites,3,index,ia)]
		 - fluxz[addr_rank1(nSites,3,indexk,ia)]);
	  fsum_local[ia] += f[ia];
	}
      }
    }
  }

  MPI_Allreduce(fsum_local, fsum, 3, MPI_DOUBLE, MPI_SUM, pe_comm());

  rv = 1.0/(L(X)*L(Y)*L(Z));

  for (ia = 0; ia < 3; ia++) {
    fsum[ia] *= rv;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = le_site_index(ic, jc, kc);
	indexj = le_site_index(ic, jc-1, kc);
	indexk = le_site_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia]
	    = - (+ fluxe[addr_rank1(nSites,3,index,ia)]
		 - fluxw[addr_rank1(nSites,3,index,ia)]
		 + fluxy[addr_rank1(nSites,3,index,ia)]
		 - fluxy[addr_rank1(nSites,3,indexj,ia)]
		 + fluxz[addr_rank1(nSites,3,index,ia)]
		 - fluxz[addr_rank1(nSites,3,indexk,ia)]);
	  f[ia] -= fsum[ia];
	}
	hydro_f_local_add(hydro, index, f);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_flux_fix_local
 *
 *  A per-plane version of the above. We know that, integrated across the
 *  area of the plane, the fluxw and fluxe contributions must be equal.
 *  Owing to the interpolation, this may not be exactly satisfied.
 *
 *  For each plane, there is therefore a correction.
 *
 *****************************************************************************/

static int phi_force_flux_fix_local(pth_t * pth, double * fluxe, double * fluxw) {

  int nlocal[3];
  int nplane;
  int ic, jc, kc, index, index1, ia, ip;

  double * fbar;     /* Local sum over plane */
  double * fcor;     /* Global correction */
  double ra;         /* Normaliser */

  MPI_Comm comm;
  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  coords_nlocal(nlocal);
  nplane = le_get_nplane_local();

  if (nplane == 0) return 0;

  comm = le_plane_comm();

  fbar = (double *) calloc(3*nplane, sizeof(double));
  fcor = (double *) calloc(3*nplane, sizeof(double));
  if (fbar == NULL) fatal("calloc(%d, fbar) failed\n", 3*nplane);
  if (fcor == NULL) fatal("calloc(%d, fcor) failed\n", 3*nplane);

  for (ip = 0; ip < nplane; ip++) { 

    ic = le_plane_location(ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = le_site_index(ic, jc, kc);
        index1 = le_site_index(ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  fbar[3*ip + ia] += - fluxe[addr_rank1(nSites,3,index,ia)]
	    + fluxw[addr_rank1(nSites,3,index1,ia)];
	}
      }
    }
  }

  MPI_Allreduce(fbar, fcor, 3*nplane, MPI_DOUBLE, MPI_SUM, comm);

  ra = 0.5/(L(Y)*L(Z));

  for (ip = 0; ip < nplane; ip++) { 

    ic = le_plane_location(ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = le_site_index(ic, jc, kc);
        index1 = le_site_index(ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nSites,3,index,ia)] += ra*fcor[3*ip + ia];
	  fluxw[addr_rank1(nSites,3,index1,ia)] -= ra*fcor[3*ip +ia];
	}
      }
    }
  }

  free(fcor);
  free(fbar);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_wallx
 *
 *  We extrapolate the stress to the wall. This is equivalent to using
 *  a one-sided gradient when we get to do the divergence.
 *
 *  The stress on the wall is recorded for accounting purposes.
 *
 *****************************************************************************/

static int phi_force_wallx(pth_t * pth, fe_t * fe, double * fluxe, double * fluxw) {

  int ic, jc, kc;
  int index, ia;
  int nlocal[3];
  double fw[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  fw[X] = 0.0;
  fw[Y] = 0.0;
  fw[Z] = 0.0;

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic,jc,kc);
	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[addr_rank1(nSites,3,index,ia)] = pth0[ia][X];
	  fw[ia] -= pth0[ia][X];
	}
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic,jc,kc);
	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nSites,3,index,ia)] = pth0[ia][X];
	  fw[ia] += pth0[ia][X];
	}
      }
    }
  }

  wall_accumulate_force(fw);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_wally
 *
 *  We extrapolate the stress to the wall. This is equivalent to using
 *  a one-sided gradient when we get to do the divergence.
 *
 *  The stress on the wall is recorded for accounting purposes.
 *
 *****************************************************************************/

static int phi_force_wally(pth_t * pth, fe_t * fe, double * fluxy) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  double fy[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  fy[X] = 0.0;
  fy[Y] = 0.0;
  fy[Z] = 0.0;

  if (cart_coords(Y) == 0) {
    jc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic, jc, kc);

	fe->func->stress(fe, index, pth0);

	/* Face flux a jc - 1 */
	index1 = le_site_index(ic, jc-1, kc);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[addr_rank1(nSites,3,index1,ia)] = pth0[ia][Y];
	  fy[ia] -= pth0[ia][Y];
	}
      }
    }
  }

  if (cart_coords(Y) == cart_size(Y) - 1) {
    jc = nlocal[Y];

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic, jc, kc);

	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[addr_rank1(nSites,3,index,ia)] = pth0[ia][Y];
	  fy[ia] += pth0[ia][Y];
	}
      }
    }
  }

  wall_accumulate_force(fy);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_wallz
 *
 *  We extrapolate the stress to the wall. This is equivalent to using
 *  a one-sided gradient when we get to do the divergence.
 *
 *  The stress on the wall is recorded for accounting purposes.
 *
 *****************************************************************************/

static int phi_force_wallz(pth_t * pth, fe_t * fe, double * fluxz) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  double fz[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  int nhalo, nSites;
  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nSites  = (nlocal[X] + 2*nhalo)*(nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

  fz[X] = 0.0;
  fz[Y] = 0.0;
  fz[Z] = 0.0;

  if (cart_coords(Z) == 0) {
    kc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = le_site_index(ic, jc, kc);

	fe->func->stress(fe, index, pth0);

	/* Face flux at kc-1 */
	index1 = le_site_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[addr_rank1(nSites,3,index1,ia)] = pth0[ia][Z];
	  fz[ia] -= pth0[ia][Z];
	}
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {
    kc = nlocal[Z];

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = le_site_index(ic, jc, kc);

	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[addr_rank1(nSites,3,index,ia)] = pth0[ia][Z];
	  fz[ia] += pth0[ia][Z];
	}
      }
    }
  }

  wall_accumulate_force(fz);

  return 0;
}
