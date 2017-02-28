/*****************************************************************************
 *
 *  phi_force.c
 *
 *  TODO:
 *  This file largely concerns computing the divergence of the
 *  stress when LE planes are present.
 *  Other material needs to be refactored. In particular, the
 *  only routine specific to phi is the phi grad mu calculation.
 *
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
 *  (c) 2011-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "kernel.h"
#include "field_grad_s.h"
#include "field_s.h"
#include "hydro_s.h"
#include "pth_s.h"
#include "timer.h"
#include "phi_force.h"
#include "phi_force_colloid.h"

static int phi_force_compute_fluxes(lees_edw_t * le, fe_t * fe, int nall,
				    double * fxe,
				    double * fxw,
				    double * fxy,
				    double * fxz);
static int phi_force_flux_divergence(cs_t * cs, hydro_t * hydro, double * fe,
				     double * fw, double * fy, double * fz);
static int phi_force_flux_fix_local(lees_edw_t * le, int nall, double * fluxe,
				    double * fluxw);
static int phi_force_flux_divergence_with_fix(cs_t * cs,
					      hydro_t * hydro, double * fe,
					      double * fw,
					      double * fy, double * fz);
static int phi_force_flux(cs_t * cs, lees_edw_t * le, fe_t * fe,
			  wall_t * wall, hydro_t * hydro);
static __host__ int phi_force_wallx(cs_t * cs, wall_t * wall, fe_t * fe, double * fxe, double * fxw);
static int phi_force_fluid_phi_gradmu(lees_edw_t * le, pth_t * pth, fe_t * fe,
				      field_t * phi,
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
 *  TODO:
 *  This routine is a bit of a mess and needs to be refactored to
 *  include the code in the main time step loop.
 *
 *****************************************************************************/

__host__ int phi_force_calculation(cs_t * cs, lees_edw_t * le, wall_t * wall,
				   pth_t * pth, fe_t * fe, map_t * map,
				   field_t * phi, hydro_t * hydro) {

  if (pth == NULL) return 0;
  if (pth->method == PTH_METHOD_NO_FORCE) return 0;
  if (hydro == NULL) return 0; 

  if (lees_edw_nplane_total(le) > 0) {
    /* Must use the flux method for LE planes */

    phi_force_flux(cs, le, fe, wall, hydro);
  }
  else {
    switch (pth->method) {
    case PTH_METHOD_DIVERGENCE:
      pth_stress_compute(pth, fe);
      if (wall_present(wall)) {
	pth_force_fluid_wall_driver(pth, hydro, map, wall);
      }
      else {
	pth_force_fluid_driver(pth, hydro);
      }
      break;
    case PTH_METHOD_GRADMU:
      phi_force_fluid_phi_gradmu(le, pth, fe, phi, hydro);
      break;
    default:
      assert(0);
    }
  }

  return 0;
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

static int phi_force_fluid_phi_gradmu(lees_edw_t * le, pth_t * pth,
				      fe_t * fe, field_t * fphi,
				      hydro_t * hydro) {

  int ic, jc, kc, icm1, icp1;
  int index0, indexm1, indexp1;
  int nhalo;
  int nlocal[3];
  int zs, ys;
  double phi, mum1, mup1;
  double force[3];

  assert(le);
  assert(fphi);
  assert(hydro);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  assert(nhalo >= 2);

  /* Memory strides */
  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, jc, kc);
	field_scalar(fphi, index0, &phi);

        indexm1 = lees_edw_index(le, icm1, jc, kc);
        indexp1 = lees_edw_index(le, icp1, jc, kc);

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
 *  TODO: the "ownership" is not very clear here. Where does it
 *        belong?
 *
 *****************************************************************************/

static int phi_force_flux(cs_t * cs, lees_edw_t * le, fe_t * fe,
			  wall_t * wall, hydro_t * hydro) {
  int n;
  int iswall[3];
  int fix_fluxes = 1;

  double * fluxe;
  double * fluxw;
  double * fluxy;
  double * fluxz;

  assert(hydro);

  wall_present_dim(wall, iswall);
  cs_nsites(cs, &n);

  fluxe = (double *) malloc(3*n*sizeof(double));
  fluxw = (double *) malloc(3*n*sizeof(double));
  fluxy = (double *) malloc(3*n*sizeof(double));
  fluxz = (double *) malloc(3*n*sizeof(double));

  if (fluxe == NULL) pe_fatal(hydro->pe, "malloc(fluxe) force failed");
  if (fluxw == NULL) pe_fatal(hydro->pe, "malloc(fluxw) force failed");
  if (fluxy == NULL) pe_fatal(hydro->pe, "malloc(fluxy) force failed");
  if (fluxz == NULL) pe_fatal(hydro->pe, "malloc(fluxz) force failed");

  phi_force_compute_fluxes(le, fe, n, fluxe, fluxw, fluxy, fluxz);

  if (iswall[X]) phi_force_wallx(cs, wall, fe, fluxe, fluxw);
  if (iswall[Y]) pe_fatal(hydro->pe, "Not allowed\n");
  if (iswall[Z]) pe_fatal(hydro->pe, "Not allowed\n");

  if (fix_fluxes || wall_present(wall)) {
    phi_force_flux_fix_local(le, n, fluxe, fluxw);
    phi_force_flux_divergence(cs, hydro, fluxe, fluxw, fluxy, fluxz);
  }
  else {
    phi_force_flux_divergence_with_fix(cs, hydro, fluxe, fluxw, fluxy, fluxz);
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


static int phi_force_compute_fluxes(lees_edw_t * le, fe_t * fe, int nall,
				    double * fluxe, double * fluxw,
				    double * fluxy, double * fluxz) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];

  double pth0[3][3];
  double pth1[3][3];

  assert(le);
  assert(fe);
  assert(fe->func->stress);

  lees_edw_nlocal(le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = lees_edw_index(le, ic, jc, kc);

	/* Compute pth at current point */
	fe->func->stress(fe, index, pth0);

	/* fluxw_a = (1/2)[P(i, j, k) + P(i-1, j, k)]_xa */
	
	index1 = lees_edw_index(le, icm1, jc, kc);

	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[addr_rank1(nall,3,index,ia)] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxe_a = (1/2)[P(i, j, k) + P(i+1, j, k)_xa */

	index1 = lees_edw_index(le, icp1, jc, kc);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nall,3,index,ia)] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxy_a = (1/2)[P(i, j, k) + P(i, j+1, k)]_ya */

	index1 = lees_edw_index(le, ic, jc+1, kc);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[addr_rank1(nall,3,index,ia)] = 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	/* fluxz_a = (1/2)[P(i, j, k) + P(i, j, k+1)]_za */

	index1 = lees_edw_index(le, ic, jc, kc+1);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[addr_rank1(nall,3,index,ia)] = 0.5*(pth1[ia][Z] + pth0[ia][Z]);
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

static int phi_force_flux_divergence(cs_t * cs, hydro_t * hydro,
				     double * fluxe, double * fluxw,
				     double * fluxy, double * fluxz) {
  int nlocal[3];
  int nsf;
  int ic, jc, kc, ia;
  int index, indexj, indexk;

  assert(cs);
  assert(hydro);
  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  cs_nlocal(cs, nlocal);
  cs_nsites(cs, &nsf);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = cs_index(cs, ic, jc, kc);
	indexj = cs_index(cs, ic, jc-1, kc);
	indexk = cs_index(cs, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  hydro->f[addr_rank1(hydro->nsite, NHDIM, index, ia)]
	    += -(+ fluxe[addr_rank1(nsf,3,index,ia)]
		 - fluxw[addr_rank1(nsf,3,index,ia)]
		 + fluxy[addr_rank1(nsf,3,index,ia)]
		 - fluxy[addr_rank1(nsf,3,indexj,ia)]
		 + fluxz[addr_rank1(nsf,3,index,ia)]
		 - fluxz[addr_rank1(nsf,3,indexk,ia)]);
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
 *  one Allreduce per call. 
 *
 *  TODO:
 *  The assert(0) indicates this routine is unused; the "local"
 *  version below is preferred.
 *
 *****************************************************************************/

static int phi_force_flux_divergence_with_fix(cs_t * cs,
					      hydro_t * hydro,
					      double * fluxe, double * fluxw,
					      double * fluxy,
					      double * fluxz) {
  int nlocal[3];
  int ic, jc, kc, index, ia;
  int indexj, indexk;
  int nsf;

  double f[3];
  double fsum_local[3];
  double fsum[3];
  double rv;
  double ltot[3];
  MPI_Comm comm;

  assert(cs);
  assert(hydro);
  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  assert(0); /* NO TEST? */

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  cs_nsites(cs, &nsf);

  for (ia = 0; ia < 3; ia++) {
    fsum_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = cs_index(cs, ic, jc, kc);
	indexj = cs_index(cs, ic, jc-1, kc);
	indexk = cs_index(cs, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia]
	    = - (+ fluxe[addr_rank1(nsf,3,index,ia)]
		 - fluxw[addr_rank1(nsf,3,index,ia)]
		 + fluxy[addr_rank1(nsf,3,index,ia)]
		 - fluxy[addr_rank1(nsf,3,indexj,ia)]
		 + fluxz[addr_rank1(nsf,3,index,ia)]
		 - fluxz[addr_rank1(nsf,3,indexk,ia)]);
	  fsum_local[ia] += f[ia];
	}
      }
    }
  }

  MPI_Allreduce(fsum_local, fsum, 3, MPI_DOUBLE, MPI_SUM, comm);

  rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

  for (ia = 0; ia < 3; ia++) {
    fsum[ia] *= rv;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = cs_index(cs, ic, jc, kc);
	indexj = cs_index(cs, ic, jc-1, kc);
	indexk = cs_index(cs, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia]
	    = - (+ fluxe[addr_rank1(nsf,3,index,ia)]
		 - fluxw[addr_rank1(nsf,3,index,ia)]
		 + fluxy[addr_rank1(nsf,3,index,ia)]
		 - fluxy[addr_rank1(nsf,3,indexj,ia)]
		 + fluxz[addr_rank1(nsf,3,index,ia)]
		 - fluxz[addr_rank1(nsf,3,indexk,ia)]);
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

static int phi_force_flux_fix_local(lees_edw_t * le, int nall,
				    double * fluxe, double * fluxw) {

  int nlocal[3];
  int nplane;
  int nhalo;
  int ic, jc, kc, index, index1, ia, ip;

  double * fbar;     /* Local sum over plane */
  double * fcor;     /* Global correction */
  double ra;         /* Normaliser */
  double ltot[3];

  MPI_Comm comm;

  assert(le);

  lees_edw_ltot(le, ltot);

  nplane = lees_edw_nplane_local(le);

  if (nplane == 0) return 0;

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  lees_edw_plane_comm(le, &comm);

  fbar = (double *) calloc(3*nplane, sizeof(double));
  fcor = (double *) calloc(3*nplane, sizeof(double));
#ifdef OLD_SHIT
  /* TODO: decide "ownership" to find pe */
  if (fbar == NULL) fatal("calloc(%d, fbar) failed\n", 3*nplane);
  if (fcor == NULL) fatal("calloc(%d, fcor) failed\n", 3*nplane);
#endif
  for (ip = 0; ip < nplane; ip++) { 

    ic = lees_edw_plane_location(le, ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = lees_edw_index(le, ic, jc, kc);
        index1 = lees_edw_index(le, ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  fbar[3*ip + ia] += - fluxe[addr_rank1(nall,3,index,ia)]
	    + fluxw[addr_rank1(nall,3,index1,ia)];
	}
      }
    }
  }

  MPI_Allreduce(fbar, fcor, 3*nplane, MPI_DOUBLE, MPI_SUM, comm);

  ra = 0.5/(ltot[Y]*ltot[Z]);

  for (ip = 0; ip < nplane; ip++) { 

    ic = lees_edw_plane_location(le, ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = lees_edw_index(le, ic, jc, kc);
        index1 = lees_edw_index(le, ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nall,3,index,ia)] += ra*fcor[3*ip + ia];
	  fluxw[addr_rank1(nall,3,index1,ia)] -= ra*fcor[3*ip +ia];
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

static __host__
int phi_force_wallx(cs_t * cs, wall_t * wall, fe_t * fe, double * fluxe,
			   double * fluxw) {

  int ic, jc, kc;
  int index, ia;
  int nsf;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double fw[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  assert(cs);
  assert(wall);
  assert(fe);
  assert(fe->func->stress);

  cs_nlocal(cs, nlocal);
  cs_nsites(cs, &nsf);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  fw[X] = 0.0;
  fw[Y] = 0.0;
  fw[Z] = 0.0;

  if (mpicoords[X] == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[addr_rank1(nsf,3,index,ia)] = pth0[ia][X];
	  fw[ia] -= pth0[ia][X];
	}
      }
    }
  }

  if (mpicoords[X] == mpisz[X] - 1) {
    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	fe->func->stress(fe, index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[addr_rank1(nsf,3,index,ia)] = pth0[ia][X];
	  fw[ia] += pth0[ia][X];
	}
      }
    }
  }

  wall_momentum_add(wall, fw);

  return 0;
}
