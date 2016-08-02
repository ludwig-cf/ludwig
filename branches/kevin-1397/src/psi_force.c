/*****************************************************************************
 *
 *  psi_force.c
 *
 *  Compute the force on the fluid originating with charge.
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga
 *    Oliver Henrich
 *
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "psi_s.h"
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
    fatal("Wrong free energy\n");
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

  coords_nlocal(nlocal);
  coords_strides(&xs, &ys, &zs);
  comm = cart_comm();

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	colloids_info_map(cinfo, index, &pc);

	/* Contribution from ionic electrostatic part
           Note: The sum over the ionic species and the
                 gradient of the electrostatic potential
                 are implicitly calculated */

	psi_rho_elec(psi, index, &rho_elec);
	psi_electric_field_d3qx(psi, index, e);

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

        index = coords_index(ic, jc, kc);

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

  coords_nlocal(nlocal);
  coords_strides(&xs, &ys, &zs);
  comm = cart_comm();

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;

  psi_nk(psi, &nk);
  assert(nk == 2); /* This routine is not completely general */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
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
	  force[X] -= rho*0.5*(musp1 - musm1)*kt;
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
	  force[Y] -= rho*0.5*(musp1 - musm1)*kt;
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
	  force[Z] -= rho*0.5*(musp1 - musm1)*kt;
	}

	/* Contribution from ionic electrostatic part
           Note: The sum over the ionic species and the
                 gradient of the electrostatic potential
                 are implicitly calculated */

	psi_rho_elec(psi, index, &rho_elec);
	psi_electric_field_d3qx(psi, index, e);

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

        index = coords_index(ic, jc, kc);

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
 *  A test routine for force via divergence of stress, allowing
 *  the stress to be computed inside the colloids.
 *
 *  The stress is to include the full electric field.
 *
 *****************************************************************************/

int psi_force_divstress(psi_t * psi, fe_t * fe, hydro_t * hydro,
			colloids_info_t * cinfo) {

  int ic, jc, kc;
  int index, index1;
  int ia;
  int nlocal[3];

  double force[3];
  double pth1[3][3];

  colloid_t * pc = NULL;

  assert(psi);
  assert(fe);
  assert(cinfo);

  coords_nlocal(nlocal);

  assert(0); /* SHIT NO TEST? */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
 
       /* Calculate divergence based on 6-pt stencil */

	index1 = coords_index(ic+1, jc, kc);
	fe->func->stress(fe, index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X]);
	}

	index1 = coords_index(ic-1, jc, kc);
        fe->func->stress(fe, index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] += 0.5*(pth1[ia][X]);
        }

        index1 = coords_index(ic, jc+1, kc);
        fe->func->stress(fe, index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] -= 0.5*(pth1[ia][Y]);
        }

        index1 = coords_index(ic, jc-1, kc);
        fe->func->stress(fe, index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] += 0.5*(pth1[ia][Y]);
        }

        index1 = coords_index(ic, jc, kc+1);
        fe->func->stress(fe, index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] -= 0.5*(pth1[ia][Z]);
        }

        index1 = coords_index(ic, jc, kc-1);
        fe->func->stress(fe, index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] += 0.5*(pth1[ia][Z]);
        }

        /* Store the force on lattice */
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


/*****************************************************************************
 *
 *  psi_force_divstress_d3qx
 *
 *  A routine for force via divergence of stress, allowing
 *  the stress to be computed inside the colloids. The 
 *  calculation of the divergence is based on the D3QX stencil
 *  with X=6, 18 or 26. The stress is to include the full 
 *  electric field.
 *
 *****************************************************************************/

int psi_force_divstress_d3qx(psi_t * psi, fe_t * fe, hydro_t * hydro,
			     map_t * map, colloids_info_t * cinfo) {

  int ic, jc, kc;
  int index, index_nb;
  int status, status_nb;
  int ia, ib;
  int nlocal[3];

  double force[3];
  double pth_nb[3][3];

  colloid_t * pc = NULL;

  int p;
  int coords[3], coords_nb[3];

  assert(psi);
  assert(cinfo);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	colloids_info_map(cinfo, index, &pc);

	coords_index_to_ijk(index, coords);

	for (ia = 0; ia < 3; ia++) {
	  force[ia] = 0.0;
	}

	/* Calculate divergence based on D3QX stencil */
	for (p = 1; p < PSI_NGRAD; p++) {

	  coords_nb[X] = coords[X] + psi_gr_cv[p][X];
	  coords_nb[Y] = coords[Y] + psi_gr_cv[p][Y];
	  coords_nb[Z] = coords[Z] + psi_gr_cv[p][Z];

	  index_nb = coords_index(coords_nb[X], coords_nb[Y], coords_nb[Z]);
	  map_status(map, index_nb, &status_nb);

	  fe->func->stress(fe, index_nb, pth_nb);

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][ib] * psi_gr_cv[p][ib];
	    }
	  }

	}

        /* Store the force on lattice */

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

/*****************************************************************************
 *
 *  psi_force_divstress_one_sided_d3qx
 *
 *  A routine for force via divergence of stress, allowing
 *  the stress to be computed inside the colloids. 
 *  This is the attempt to use one-sided derivatives at the
 *  surface of the particles.
 *  The calculation of the divergence is based on the D3QX stencil
 *  with X=6, 18 or 26. The stress is to include the full 
 *  electric field. 
 *
 *  NOTE
 *  This routine has not been refactored as no test exists. The
 *  refactoring has retained hardwired references to fe_electro
 *  (only).
 *
 *****************************************************************************/

int psi_force_divstress_one_sided_d3qx(psi_t * psi, hydro_t * hydro, map_t * map, colloids_info_t * cinfo) {

  int ic, jc, kc;
  int index, index_nb, index1, index2;
  int status, status_nb;
  int ia, ib;
  int nlocal[3];
  int p;
  int coords[3], coords_nb[3], coords1[3], coords2[3];

  double force[3];
  double pth1[3][3], pth2[3][3];
  double pth[3][3], pth_nb[3][3];

  fe_electro_t * fe = NULL; /* See note above */
  colloid_t * pc = NULL;

  assert(psi);
  assert(cinfo);

  coords_nlocal(nlocal);

  assert(0); /* SHIT NO TEST? */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	colloids_info_map(cinfo, index, &pc);

	coords_index_to_ijk(index, coords);

	for (ia = 0; ia < 3; ia++) {
	  force[ia] = 0.0;
	}

	/* Calculate divergence based on D3QX stencil */
	for (p = 1; p < PSI_NGRAD; p++) {

	  coords_nb[X] = coords[X] + psi_gr_cv[p][X];
	  coords_nb[Y] = coords[Y] + psi_gr_cv[p][Y];
	  coords_nb[Z] = coords[Z] + psi_gr_cv[p][Z];

	  index_nb = coords_index(coords_nb[X], coords_nb[Y], coords_nb[Z]);
	  map_status(map, index_nb, &status_nb);

          if (status != MAP_FLUID) {

	    fe_electro_stress_ex(fe, index_nb, pth_nb);

	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][ib] * psi_gr_cv[p][ib];
	      }
	    }

	  }

          if (status == MAP_FLUID && status_nb == MAP_FLUID) {

	    fe_electro_stress(fe, index_nb, pth_nb);

	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][X] * psi_gr_cv[p][X];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][Y] * psi_gr_cv[p][Y];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][Z] * psi_gr_cv[p][Z];
	    }
	  }

          if (status == MAP_FLUID && status_nb != MAP_FLUID) {

	    /* Current site r */
	    fe_electro_stress(fe, index, pth);

	    /* Site r - cv */
	    coords1[X] = coords[X] - psi_gr_cv[p][X];
	    coords1[Y] = coords[Y] - psi_gr_cv[p][Y];
	    coords1[Z] = coords[Z] - psi_gr_cv[p][Z];

	    index1 = coords_index(coords1[X], coords1[Y], coords1[Z]);

	    fe_electro_stress(fe, index1, pth1);

	    /* Subtract the above 'fluid' half of the incomplete two-point formula. */
	    /* Note: subtracting means adding here because of inverse lattice vectors. */
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth1[ia][X] * psi_gr_cv[p][X];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth1[ia][Y] * psi_gr_cv[p][Y];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth1[ia][Z] * psi_gr_cv[p][Z];
	    }

	    /* Site r - 2*cv */
	    coords2[X] = coords[X] - 2*psi_gr_cv[p][X];
	    coords2[Y] = coords[Y] - 2*psi_gr_cv[p][Y];
	    coords2[Z] = coords[Z] - 2*psi_gr_cv[p][Z];

	    index2 = coords_index(coords2[X], coords2[Y], coords2[Z]);

	    fe_electro_stress(fe, index2, pth2);

	    /* Use one-sided derivative instead */
	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * 
			(3.0*pth[ia][ib] - 4.0*pth1[ia][ib] + 1.0*pth2[ia][ib]) 
				* psi_gr_rnorm[p]* psi_gr_cv[p][ib];
	      }
	    }

	  }


	}


        /* Store the force on lattice */
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
