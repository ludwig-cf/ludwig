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
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "free_energy.h"
#include "wall.h"
#include "advection_s.h"
#include "phi_force.h"

static int phi_force_calculation_fluid(le_t * le, hydro_t * hydro);
static int phi_force_compute_fluxes(advflux_t * flux);
static int phi_force_flux_divergence(advflux_t * flux, hydro_t * hydro);
static int phi_force_flux_fix_local(advflux_t * flux);
static int phi_force_flux_divergence_fix(advflux_t * flux, hydro_t * hydro);
static int phi_force_flux(le_t * le, hydro_t * hydro);
static int phi_force_wallx(advflux_t * flux);
static int phi_force_wally(advflux_t * flux);
static int phi_force_wallz(advflux_t * flux);

static int phi_force_fluid_phi_gradmu(le_t * le, field_t * phi,
				      hydro_t * hydro);

static int force_required_ = 1;
static int force_divergence_ = 1;

/*****************************************************************************
 *
 *  phi_force_required_set
 *
 *****************************************************************************/

int phi_force_required_set(const int flag) {

  force_required_ = flag;
  return 0;
}

/*****************************************************************************
 *
 *  phi_force_required
 *
 *****************************************************************************/

int phi_force_required(int * flag) {

  assert(flag);

  *flag = force_required_;

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_divergence_set
 *
 *****************************************************************************/

int phi_force_divergence_set(const int flag) {

  force_divergence_ = flag;
  return 0;
}

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

int phi_force_calculation(le_t * le, field_t * phi, hydro_t * hydro) {

  int nplane;

  if (force_required_ == 0) return 0;
  if (hydro == NULL) return 0;

  assert(le);
  le_nplane_total(le, &nplane);

  if (nplane > 0 || wall_present()) {
    /* Must use the flux method for LE planes */
    /* Also convenient for plane walls */
    phi_force_flux(le, hydro);
  }
  else {
    if (force_divergence_) {
      phi_force_calculation_fluid(le, hydro);
   }
    else {
      assert(phi);
      phi_force_fluid_phi_gradmu(le, phi, hydro);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *  using a simple six-point stencil.
 *
 *  Side effect: increments the force at each local lattice site in
 *  preparation for the collision stage.
 *
 *****************************************************************************/

/* PENDING not really le planes */

static int phi_force_calculation_fluid(le_t * le, hydro_t * hydro) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  void (*chemical_stress)(const int index, double s[3][3]);

  chemical_stress = fe_chemical_stress_function();

  assert(le);
  assert(hydro);

  le_nlocal(le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(le, ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = le_site_index(le, icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}
	index1 = le_site_index(le, icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}
	index1 = le_site_index(le, ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	index1 = le_site_index(le, ic, jc-1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	index1 = le_site_index(le, ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}
	index1 = le_site_index(le, ic, jc, kc-1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index, force);

	/* Next site */
      }
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

/* PENDING not really le planes */

static int phi_force_fluid_phi_gradmu(le_t * le, field_t * fphi,
				      hydro_t * hydro) {

  int ic, jc, kc, icm1, icp1;
  int index0, indexm1, indexp1;
  int nhalo;
  int nlocal[3];
  int zs, ys, xs;
  double phi, mum1, mup1;
  double force[3];

  double (* chemical_potential)(const int index, const int nop);

  assert(le);
  assert(fphi);
  assert(hydro);

  le_nhalo(le, &nhalo);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);
  assert(nhalo >= 2);

  chemical_potential = fe_chemical_potential_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(le, ic, jc, kc);
	field_scalar(fphi, index0, &phi);

        indexm1 = le_site_index(le, icm1, jc, kc);
        indexp1 = le_site_index(le, icp1, jc, kc);

        mum1 = chemical_potential(indexm1, 0);
        mup1 = chemical_potential(indexp1, 0);

        force[X] = -phi*0.5*(mup1 - mum1);

        mum1 = chemical_potential(index0 - ys, 0);
        mup1 = chemical_potential(index0 + ys, 0);

        force[Y] = -phi*0.5*(mup1 - mum1);

        mum1 = chemical_potential(index0 - zs, 0);
        mup1 = chemical_potential(index0 + zs, 0);

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

static int phi_force_flux(le_t * le, hydro_t * hydro) {

  int fix_fluxes = 1;
  advflux_t * fluxes = NULL;

  assert(le);
  assert(hydro);

  advflux_create(le, 3, &fluxes);

  phi_force_compute_fluxes(fluxes);

  if (wall_at_edge(X)) phi_force_wallx(fluxes);
  if (wall_at_edge(Y)) phi_force_wally(fluxes);
  if (wall_at_edge(Z)) phi_force_wallz(fluxes);

  if (fix_fluxes || wall_present()) {
    phi_force_flux_fix_local(fluxes);
    phi_force_flux_divergence(fluxes, hydro);
  }
  else {
    phi_force_flux_divergence_fix(fluxes, hydro);
  }

  advflux_free(fluxes);

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_compute_fluxes
 *
 *  Linearly interpolate the chemical stress to the cell faces to get
 *  the momentum fluxes. 
 *
 *****************************************************************************/

static int phi_force_compute_fluxes(advflux_t * flux) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(flux);

  le_nlocal(flux->le, nlocal);
  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(flux->le, ic, -1);
    icp1 = le_index_real_to_buffer(flux->le, ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = le_site_index(flux->le, ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* fluxw_a = (1/2)[P(i, j, k) + P(i-1, j, k)]_xa */

	index1 = le_site_index(flux->le, icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  flux->fw[3*index + ia] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxe_a = (1/2)[P(i, j, k) + P(i+1, j, k)_xa */

	index1 = le_site_index(flux->le, icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  flux->fe[3*index + ia] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxy_a = (1/2)[P(i, j, k) + P(i, j+1, k)]_ya */

	index1 = le_site_index(flux->le, ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  flux->fy[3*index + ia] = 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	
	/* fluxz_a = (1/2)[P(i, j, k) + P(i, j, k+1)]_za */

	index1 = le_site_index(flux->le, ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  flux->fz[3*index + ia] = 0.5*(pth1[ia][Z] + pth0[ia][Z]);
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

static int phi_force_flux_divergence(advflux_t * flux, hydro_t * hydro) {

  int ic, jc, kc, ia;
  int index, indexj, indexk;
  int nlocal[3];
  double f[3];

  assert(flux);
  assert(hydro);

  le_nlocal(flux->le, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = le_site_index(flux->le, ic, jc, kc);
	indexj = le_site_index(flux->le, ic, jc-1, kc);
	indexk = le_site_index(flux->le, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia] = - (flux->fe[3*index + ia] - flux->fw[3*index + ia]
		   + flux->fy[3*index + ia] - flux->fy[3*indexj + ia]
		   + flux->fz[3*index + ia] - flux->fz[3*indexk + ia]);
	}

	hydro_f_local_add(hydro, index, f);

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_flux_divergence_fix
 *
 *  Take the diverence of the momentum fluxes to get a force on the
 *  fluid site.
 *
 *  It is intended that these fluxes are uncorrected, and that a
 *  global constraint on the total force is enforced. This costs
 *  one Allreduce in pe_comm() per call. 
 *
 *****************************************************************************/

static int phi_force_flux_divergence_fix(advflux_t * flux, hydro_t * hydro) {

  int nlocal[3];
  int ic, jc, kc, index, ia;
  int indexj, indexk;
  double f[3];
  double ltot[3];

  double fsum_local[3];
  double fsum[3];
  double rv;

  assert(flux);
  assert(hydro);

  le_ltot(flux->le, ltot);
  le_nlocal(flux->le, nlocal);

  for (ia = 0; ia < 3; ia++) {
    fsum_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = le_site_index(flux->le, ic, jc, kc);
	indexj = le_site_index(flux->le, ic, jc-1, kc);
	indexk = le_site_index(flux->le, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia] = - (flux->fe[3*index + ia] - flux->fw[3*index + ia]
		   + flux->fy[3*index + ia] - flux->fy[3*indexj + ia]
		   + flux->fz[3*index + ia] - flux->fz[3*indexk + ia]);
	  fsum_local[ia] += f[ia];
	}
      }
    }
  }

  MPI_Allreduce(fsum_local, fsum, 3, MPI_DOUBLE, MPI_SUM, pe_comm());

  rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

  for (ia = 0; ia < 3; ia++) {
    fsum[ia] *= rv;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index =  le_site_index(flux->le, ic, jc, kc);
	indexj = le_site_index(flux->le, ic, jc-1, kc);
	indexk = le_site_index(flux->le, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  f[ia] = - (flux->fe[3*index + ia] - flux->fw[3*index + ia]
		   + flux->fy[3*index + ia] - flux->fy[3*indexj + ia]
		   + flux->fz[3*index + ia] - flux->fz[3*indexk + ia]);
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

static int phi_force_flux_fix_local(advflux_t * flux) {

  int nlocal[3];
  int nplane;
  int ic, jc, kc, index, index1, ia, ip;

  double * fbar;     /* Local sum over plane */
  double * fcor;     /* Global correction */
  double ra;         /* Normaliser */
  double ltot[3];

  MPI_Comm comm;

  assert(flux);

  le_ltot(flux->le, ltot);
  le_nlocal(flux->le, nlocal);
  le_nplane_local(flux->le, &nplane);

  if (nplane == 0) return 0;

  le_plane_comm(flux->le, &comm);

  fbar = (double *) calloc(3*nplane, sizeof(double));
  fcor = (double *) calloc(3*nplane, sizeof(double));
  if (fbar == NULL) fatal("calloc(%d, fbar) failed\n", 3*nplane);
  if (fcor == NULL) fatal("calloc(%d, fcor) failed\n", 3*nplane);

  for (ip = 0; ip < nplane; ip++) { 

    ic = le_plane_location(flux->le, ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = le_site_index(flux->le, ic, jc, kc);
        index1 = le_site_index(flux->le, ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  fbar[3*ip + ia] += - flux->fe[3*index + ia] + flux->fw[3*index1 + ia];
	}
      }
    }
  }

  MPI_Allreduce(fbar, fcor, 3*nplane, MPI_DOUBLE, MPI_SUM, comm);

  ra = 0.5/(ltot[Y]*ltot[Z]);

  for (ip = 0; ip < nplane; ip++) { 

    ic = le_plane_location(flux->le, ip);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index  = le_site_index(flux->le, ic, jc, kc);
        index1 = le_site_index(flux->le, ic + 1, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  flux->fe[3*index  + ia] += ra*fcor[3*ip + ia];
	  flux->fw[3*index1 + ia] -= ra*fcor[3*ip + ia];
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

static int phi_force_wallx(advflux_t * flux) {

  int ic, jc, kc;
  int index, ia;
  int nlocal[3];
  int cartsz[3];
  int cartcoords[3];
  double fw[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(flux);

  le_cartsz(flux->le, cartsz);
  le_cart_coords(flux->le, cartcoords);
  le_nlocal(flux->le, nlocal);

  chemical_stress = fe_chemical_stress_function();

  fw[X] = 0.0;
  fw[Y] = 0.0;
  fw[Z] = 0.0;

  if (cartcoords[X] == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(flux->le, ic,jc,kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  flux->fw[3*index + ia] = pth0[ia][X];
	  fw[ia] -= pth0[ia][X];
	}
      }
    }
  }

  if (cartcoords[X] == cartsz[X] - 1) {
    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(flux->le, ic,jc,kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  flux->fe[3*index + ia] = pth0[ia][X];
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

static int phi_force_wally(advflux_t * flux) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  int cartsz[3];
  int cartcoords[3];
  double fy[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(flux);

  le_cartsz(flux->le, cartsz);
  le_cart_coords(flux->le, cartcoords);
  le_nlocal(flux->le, nlocal);

  chemical_stress = fe_chemical_stress_function();

  fy[X] = 0.0;
  fy[Y] = 0.0;
  fy[Z] = 0.0;

  if (cartcoords[Y] == 0) {
    jc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(flux->le, ic, jc, kc);
	chemical_stress(index, pth0);

	/* Face flux a jc - 1 */
	index1 = le_site_index(flux->le, ic, jc-1, kc);

	for (ia = 0; ia < 3; ia++) {
	  flux->fy[3*index1 + ia] = pth0[ia][Y];
	  fy[ia] -= pth0[ia][Y];
	}
      }
    }
  }

  if (cartcoords[Y] == cartsz[Y] - 1) {
    jc = nlocal[Y];

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(flux->le, ic, jc, kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  flux->fy[3*index + ia] = pth0[ia][Y];
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

static int phi_force_wallz(advflux_t * flux) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  int cartsz[3];
  int cartcoords[3];
  double fz[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(flux);
  le_cartsz(flux->le, cartsz);
  le_cart_coords(flux->le, cartcoords);
  le_nlocal(flux->le, nlocal);

  chemical_stress = fe_chemical_stress_function();

  fz[X] = 0.0;
  fz[Y] = 0.0;
  fz[Z] = 0.0;

  if (cartcoords[Z] == 0) {
    kc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = le_site_index(flux->le, ic, jc, kc);
	chemical_stress(index, pth0);

	/* Face flux at kc-1 */
	index1 = le_site_index(flux->le, ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  flux->fz[3*index1 + ia] = pth0[ia][Z];
	  fz[ia] -= pth0[ia][Z];
	}
      }
    }
  }

  if (cartcoords[Z] == cartsz[Z] - 1) {
    kc = nlocal[Z];

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = le_site_index(flux->le, ic, jc, kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  flux->fz[3*index + ia] = pth0[ia][Z];
	  fz[ia] += pth0[ia][Z];
	}
      }
    }
  }

  wall_accumulate_force(fz);

  return 0;
}
