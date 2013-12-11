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

#include "pe.h"
#include "coords.h"
#include "hydro.h"
#include "field.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "wall.h"
#include "phi_force_stress.h"

static int phi_force_calculation_fluid(hydro_t * hydro, double dt);
static int phi_force_compute_fluxes(double * fe, double * fw, double * fy,
				    double * fz);
static int phi_force_flux_divergence(hydro_t * hydro, double * fe,
				     double * fw, double * fy, double * fz);
static int phi_force_flux_fix_local(double * fluxe, double * fluxw);
static int phi_force_flux_divergence_with_fix(hydro_t * hydro, double * fe,
					      double * fw,
					      double * fy, double * fz);
static int phi_force_flux(hydro_t * hydro);
static int phi_force_wallx(double * fe, double * fw);
static int phi_force_wally(double * fy);
static int phi_force_wallz(double * fz);

static int phi_force_fluid_phi_gradmu(field_t * phi, hydro_t * hydro, double dt);

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

int phi_force_calculation(field_t * phi, hydro_t * hydro, double dt) {

  if (force_required_ == 0) return 0;
  if (hydro == NULL) return 0;

  if (le_get_nplane_total() > 0 || wall_present()) {
    /* Must use the flux method for LE planes */
    /* Also convenient for plane walls */
    phi_force_flux(hydro);
  }
  else {
    if (force_divergence_) {
      phi_force_calculation_fluid(hydro, dt);
   }
    else {
      assert(phi);
      phi_force_fluid_phi_gradmu(phi, hydro, dt);
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

static int phi_force_calculation_fluid(hydro_t * hydro, double dt) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  void (* chemical_stress)(const int index, double s[3][3]);

  assert(hydro);

  coords_nlocal(nlocal);

  phi_force_stress_allocate();
  phi_force_stress_compute();

  chemical_stress = phi_force_stress;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = le_site_index(icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X])*dt;
	}
	index1 = le_site_index(icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X])*dt;
	}
	index1 = le_site_index(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y])*dt;
	}
	index1 = le_site_index(ic, jc-1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y])*dt;
	}
	index1 = le_site_index(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z])*dt;
	}
	index1 = le_site_index(ic, jc, kc-1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z])*dt;
	}

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index, force);

	/* Next site */
      }
    }
  }

  phi_force_stress_free();

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

static int phi_force_fluid_phi_gradmu(field_t * fphi, hydro_t * hydro, double dt) {

  int ic, jc, kc, icm1, icp1;
  int index0, indexm1, indexp1;
  int nhalo;
  int nlocal[3];
  int zs, ys;
  double phi, mum1, mup1;
  double force[3];

  double (* chemical_potential)(const int index, const int nop);

  assert(fphi);
  assert(hydro);
  /* Could check nf == 1 sclar field. */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  assert(nhalo >= 2);

  /* Memory strides */
  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;

  chemical_potential = fe_chemical_potential_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);
	field_scalar(fphi, index0, &phi);

        indexm1 = le_site_index(icm1, jc, kc);
        indexp1 = le_site_index(icp1, jc, kc);

        mum1 = chemical_potential(indexm1, 0);
        mup1 = chemical_potential(indexp1, 0);

        force[X] = -phi*0.5*(mup1 - mum1)*dt;

        mum1 = chemical_potential(index0 - ys, 0);
        mup1 = chemical_potential(index0 + ys, 0);

        force[Y] = -phi*0.5*(mup1 - mum1)*dt;

        mum1 = chemical_potential(index0 - zs, 0);
        mup1 = chemical_potential(index0 + zs, 0);

        force[Z] = -phi*0.5*(mup1 - mum1)*dt;

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

static int phi_force_flux(hydro_t * hydro) {

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

  phi_force_compute_fluxes(fluxe, fluxw, fluxy, fluxz);

  if (wall_at_edge(X)) phi_force_wallx(fluxe, fluxw);
  if (wall_at_edge(Y)) phi_force_wally(fluxy);
  if (wall_at_edge(Z)) phi_force_wallz(fluxz);

  if (fix_fluxes || wall_present()) {
    phi_force_flux_fix_local(fluxe, fluxw);
    phi_force_flux_divergence(hydro, fluxe, fluxw, fluxy, fluxz);
  }
  else {
    phi_force_flux_divergence_with_fix(hydro, fluxe, fluxw, fluxy, fluxz);
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
 *****************************************************************************/

static int phi_force_compute_fluxes(double * fluxe, double * fluxw,
				    double * fluxy, double * fluxz) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* fluxw_a = (1/2)[P(i, j, k) + P(i-1, j, k)]_xa */
	
	index1 = le_site_index(icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxw[3*index + ia] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxe_a = (1/2)[P(i, j, k) + P(i+1, j, k)_xa */

	index1 = le_site_index(icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxe[3*index + ia] = 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	/* fluxy_a = (1/2)[P(i, j, k) + P(i, j+1, k)]_ya */

	index1 = le_site_index(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxy[3*index + ia] = 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	
	/* fluxz_a = (1/2)[P(i, j, k) + P(i, j, k+1)]_za */

	index1 = le_site_index(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxz[3*index + ia] = 0.5*(pth1[ia][Z] + pth0[ia][Z]);
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

static int phi_force_flux_divergence(hydro_t * hydro, double * fluxe,
				     double * fluxw,
				     double * fluxy, double * fluxz) {
  int nlocal[3];
  int ic, jc, kc, ia;
  int index, indexj, indexk;
  double f[3];

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
	  f[ia] = - (fluxe[3*index + ia] - fluxw[3*index + ia]
		     + fluxy[3*index + ia] - fluxy[3*indexj + ia]
		     + fluxz[3*index + ia] - fluxz[3*indexk + ia]);
	}

	hydro_f_local_add(hydro, index, f);

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

static int phi_force_flux_divergence_with_fix(hydro_t * hydro,
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

  assert(hydro);
  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

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
	  f[ia] = - (fluxe[3*index + ia] - fluxw[3*index + ia]
		     + fluxy[3*index + ia] - fluxy[3*indexj + ia]
		     + fluxz[3*index + ia] - fluxz[3*indexk + ia]);
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
	  f[ia] = - (fluxe[3*index + ia] - fluxw[3*index + ia]
		     + fluxy[3*index + ia] - fluxy[3*indexj + ia]
		     + fluxz[3*index + ia] - fluxz[3*indexk + ia]);
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

static int phi_force_flux_fix_local(double * fluxe, double * fluxw) {

  int nlocal[3];
  int nplane;
  int ic, jc, kc, index, index1, ia, ip;

  double * fbar;     /* Local sum over plane */
  double * fcor;     /* Global correction */
  double ra;         /* Normaliser */

  MPI_Comm comm;

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
	  fbar[3*ip + ia] += - fluxe[3*index + ia] + fluxw[3*index1 + ia];
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
	  fluxe[3*index  + ia] += ra*fcor[3*ip + ia];
	  fluxw[3*index1 + ia] -= ra*fcor[3*ip + ia];
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

static int phi_force_wallx(double * fluxe, double * fluxw) {

  int ic, jc, kc;
  int index, ia;
  int nlocal[3];
  double fw[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);

  chemical_stress = fe_chemical_stress_function();

  fw[X] = 0.0;
  fw[Y] = 0.0;
  fw[Z] = 0.0;

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic,jc,kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[3*index + ia] = pth0[ia][X];
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
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[3*index + ia] = pth0[ia][X];
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

static int phi_force_wally(double * fluxy) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  double fy[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);

  chemical_stress = fe_chemical_stress_function();

  fy[X] = 0.0;
  fy[Y] = 0.0;
  fy[Z] = 0.0;

  if (cart_coords(Y) == 0) {
    jc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = le_site_index(ic, jc, kc);
	chemical_stress(index, pth0);

	/* Face flux a jc - 1 */
	index1 = le_site_index(ic, jc-1, kc);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[3*index1 + ia] = pth0[ia][Y];
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
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxy[3*index + ia] = pth0[ia][Y];
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

static int phi_force_wallz(double * fluxz) {

  int ic, jc, kc;
  int index, index1, ia;
  int nlocal[3];
  double fz[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress at fluid point next to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);

  chemical_stress = fe_chemical_stress_function();

  fz[X] = 0.0;
  fz[Y] = 0.0;
  fz[Z] = 0.0;

  if (cart_coords(Z) == 0) {
    kc = 1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = le_site_index(ic, jc, kc);
	chemical_stress(index, pth0);

	/* Face flux at kc-1 */
	index1 = le_site_index(ic, jc, kc-1);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[3*index1 + ia] = pth0[ia][Z];
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
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxz[3*index + ia] = pth0[ia][Z];
	  fz[ia] += pth0[ia][Z];
	}
      }
    }
  }

  wall_accumulate_force(fz);

  return 0;
}
