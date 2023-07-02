/*****************************************************************************
 *
 *  psi_force.c
 *
 *  Compute the force on the fluid originating with charge.
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga
 *    Oliver Henrich
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "fe_electro.h"
#include "fe_electro_symmetric.h"
#include "psi_force.h"
#include "psi_gradients.h"

int psi_force_gradmu_e(psi_t * psi, fe_t * fe, hydro_t * hydro,
		       colloids_info_t * cinfo);
int psi_force_gradmu_es(psi_t * psi, fe_t * fe, field_t * phi, hydro_t * hydro,
			colloids_info_t * cinfo);

/*****************************************************************************
 *
 *  psi_force_gradmu
 *
 *  This routine computes the force on the fluid via the gradient
 *  of the chemical potential.
 *
 *****************************************************************************/

int psi_force_gradmu(psi_t * psi, fe_t * fe, field_t * phi,
		     hydro_t * hydro,
		     map_t * map, colloids_info_t * cinfo) {

  assert(fe);

  switch (fe->id) {
  case FE_ELECTRO:
    psi_force_gradmu_e(psi, fe, hydro, cinfo);
    break;
  case FE_ELECTRO_SYMMETRIC:
    psi_force_gradmu_es(psi, fe, phi, hydro, cinfo);
   break;
  default:
    pe_fatal(psi->pe, "Wrong free energy\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_gradmu_e
 *
 *  The first of two versions, this one for FE_ELECTRO.
 *  There is some repetition of code which could be rationalised.
 *
 *  If hydro is NULL, there is no force on the fluid, but there
 *  can be a force on the colloids.
 *
 *****************************************************************************/

int psi_force_gradmu_e(psi_t * psi, fe_t * fe, hydro_t * hydro,
		       colloids_info_t * cinfo) {

  int ic, jc, kc;
  int ia;
  int nlocal[3];
  int index;
  int xs, ys, zs;       /* Coordinate strides */
  double rho_elec;      /* Species and electric charge density */
  double e[3];          /* Total electric field */
  double kt, eunit, reunit;
  double force[3];
  /* Cummulative forces for momentum correction */
  double flocal[4] = {0.0, 0.0, 0.0, 0.0};
  double fsum[4];

  physics_t * phys = NULL;
  MPI_Comm comm;

  colloid_t * pc = NULL;

  assert(fe);
  assert(psi);
  assert(cinfo);

  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);
  cs_cart_comm(psi->cs, &comm);

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);
	colloids_info_map(cinfo, index, &pc);

	/* Contribution from ionic electrostatic part
           Note: The sum over the ionic species and the
                 gradient of the electrostatic potential
                 are implicitly calculated */

	psi_rho_elec(psi, index, &rho_elec);
	psi_electric_field(psi, index, e);

	for (ia = 0; ia < 3; ia++) {
	  e[ia] *= kt*reunit;
	  force[ia] = rho_elec*e[ia];
	}

	/* If solid, accumulate contribution to colloid;
	   otherwise to fluid node */

	if (pc) {
	  pc->force[X] += force[X];
	  pc->force[Y] += force[Y];
	  pc->force[Z] += force[Z];
	}
	else {
	  if (hydro) hydro_f_local_add(hydro, index, force);
	  flocal[3] += 1.0;
	}

	/* Accumulate contribution to total force on system */

	flocal[X] += force[X];
	flocal[Y] += force[Y];
	flocal[Z] += force[Z];

      }
    }
  }

  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  fsum[X] /= fsum[3];
  fsum[Y] /= fsum[3];
  fsum[Z] /= fsum[3];

  /* Now actually compute the force on the fluid with the correction
     (based on number of fluid nodes) and store */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);

	colloids_info_map(cinfo, index, &pc);
	if (pc) continue;

        force[X] = - fsum[X];
        force[Y] = - fsum[Y];
        force[Z] = - fsum[Z];

	if (hydro) hydro_f_local_add(hydro, index, force);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_gradmu_es
 *
 *  This for FE_ELECTRO_SYMMETRIC, including the solvation and
 *  composition-dependent terms.
 *
 *  Note: The ionic solvation free energy difference is in units 
 *        of kt and must be dressed for the force calculation.
 *
 *****************************************************************************/

int psi_force_gradmu_es(psi_t * psi, fe_t * fe, field_t * phi, hydro_t * hydro,
			colloids_info_t * cinfo) {

  int ic, jc, kc;
  int in, nk;
  int ia;
  int nlocal[3];
  int index;
  int xs, ys, zs;       /* Coordinate strides */
  double rho, rho_elec; /* Species and electric charge density */
  double e[3];          /* Total electric field */
  double muphim1, muphip1, musm1, musp1;
  double phi0;          /* Compositional order parameter */
  double kt, eunit, reunit;
  double force[3];
  /* Cummulative forces for momentum correction */
  double flocal[4] = {0.0, 0.0, 0.0, 0.0};
  double fsum[4];

  physics_t * phys = NULL;
  MPI_Comm comm;

  colloid_t * pc = NULL;

  assert(fe);
  assert(psi);
  assert(phi);
  assert(cinfo);

  cs_nlocal(psi->cs, nlocal);
  cs_strides(psi->cs, &xs, &ys, &zs);
  cs_cart_comm(psi->cs, &comm);

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  psi_nk(psi, &nk);
  assert(nk == 2); /* This routine is not completely general */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);
	colloids_info_map(cinfo, index, &pc);

	/* X-direction */

	field_scalar(phi, index, &phi0);

	/* Contribution from composition part */
	fe->func->mu(fe, index - xs, &muphim1);
	fe->func->mu(fe, index + xs, &muphip1);

	force[X] = -phi0*0.5*(muphip1 - muphim1);

	/* Contribution from ionic solvation part */
	for (in = 0; in < nk; in++) {
	  psi_rho(psi, index, in, &rho);
	  fe->func->mu_solv(fe, index - xs, in, &musm1);
	  fe->func->mu_solv(fe, index + xs, in, &musp1);
	  force[X] -= rho*0.5*(musp1 - musm1);
	}

	/* Y-direction */

	/* Contribution from composition part */
	fe->func->mu(fe, index - ys, &muphim1);
	fe->func->mu(fe, index + ys, &muphip1);
	force[Y] = -phi0*0.5*(muphip1 - muphim1);

	/* Contribution from ionic solvation part */

	for (in = 0; in < nk; in++) {
	  psi_rho(psi, index, in, &rho); 
	  fe->func->mu_solv(fe, index - ys, in, &musm1);
	  fe->func->mu_solv(fe, index + ys, in, &musp1);
	  force[Y] -= rho*0.5*(musp1 - musm1);
	}

	/* Z-direction */
	/* Contribution from composition part */

	fe->func->mu(fe, index - zs, &muphim1);
	fe->func->mu(fe, index + zs, &muphip1);
	force[Z] = -phi0*0.5*(muphip1 - muphim1);

	/* Contribution from ionic solvation part */
	for (in = 0; in < nk; in++) {
	  psi_rho(psi, index, in, &rho); 
	  fe->func->mu_solv(fe, index - zs, in, &musm1);
	  fe->func->mu_solv(fe, index + zs, in, &musp1);
	  force[Z] -= rho*0.5*(musp1 - musm1);
	}

	/* Contribution from ionic electrostatic part
           Note: The sum over the ionic species and the
                 gradient of the electrostatic potential
                 are implicitly calculated */

	psi_rho_elec(psi, index, &rho_elec);
	psi_electric_field(psi, index, e);

	for (ia = 0; ia < 3; ia++) {
	  e[ia] *= kt*reunit;
	  force[ia] += rho_elec*e[ia];
	}

	/* If solid, accumulate contribution to colloid;
	   otherwise to fluid node */

	if (pc) {

	  pc->force[X] += force[X];
	  pc->force[Y] += force[Y];
	  pc->force[Z] += force[Z];

	}
	else {
	  if (hydro) hydro_f_local_add(hydro, index, force);
	  flocal[3] += 1.0;
	}

	/* Accumulate contribution to total force on system */

	flocal[X] += force[X];
	flocal[Y] += force[Y];
	flocal[Z] += force[Z];

      }
    }
  }

  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  fsum[X] /= fsum[3];
  fsum[Y] /= fsum[3];
  fsum[Z] /= fsum[3];

  /* Now actually compute the force on the fluid with the correction
     (based on number of fluid nodes) and store */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);

	colloids_info_map(cinfo, index, &pc);
	if (pc) continue;

        force[X] = - fsum[X];
        force[Y] = - fsum[Y];
        force[Z] = - fsum[Z];

	if (hydro) hydro_f_local_add(hydro, index, force);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_divstress
 *
 *  A routine for force via divergence of stress, allowing
 *  the stress to be computed inside the colloids.
 *
 *  The stress is to include the full electric field.
 *
 *****************************************************************************/

int psi_force_divstress(psi_t * psi, fe_t * fe, hydro_t * hydro,
			colloids_info_t * cinfo) {

  int nlocal[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  assert(psi);
  assert(cinfo);

  cs = psi->cs;
  s  = psi->stencil;
  assert(cs);
  assert(s);

  cs_nlocal(cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	double force[3] = {0};
	colloid_t * pc = NULL;

	colloids_info_map(cinfo, index, &pc);

	/* Calculate divergence based on the stencil */
	for (int p = 1; p < s->npoints; p++) {

	  int8_t cx = s->cv[p][X];
	  int8_t cy = s->cv[p][Y];
	  int8_t cz = s->cv[p][Z];
	  int index1 = cs_index(cs, ic + cx, jc + cy, kc + cz);
	  double pth[3][3] = {0};

	  fe->func->stress(fe, index1, pth);

	  for (int ia = 0; ia < 3; ia++) {
	    for (int ib = 0; ib < 3; ib++) {
	      force[ia] -= s->wgradients[p]*pth[ia][ib]*s->cv[p][ib];
	    }
	  }
	}

        /* Store the force on the colloid or on the lattice */

	if (pc) {
	  pc->force[X] += force[X];
	  pc->force[Y] += force[Y];
	  pc->force[Z] += force[Z];
	}
	else {
	  hydro_f_local_add(hydro, index, force);
	}

      }
    }
  }

  return 0;
}
