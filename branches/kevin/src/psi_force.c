/*****************************************************************************
 *
 *  psi_force.c
 *
 *  Compute the force on the fluid originating with charge.
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013 The University of Edinburgh
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
#include "psi_s.h"
#include "fe_electro.h"
#include "fe_electro_symmetric.h"
#include "psi_force.h"
#include "psi_gradients.h"

static int psi_force_divergence_ = 1;

/*****************************************************************************
 *
 *  psi_force_grad_mu
 *
 *  The force density is
 *    f_a = - \sum_k rho_k grad_a mu^ex_k
 *  where mu_ex is the excess chemical potential (above ideal gas part).
 *  So
 *    f_a = - \sum_k rho_k grad_a z_k e psi
 *        = - rho_el grad_a psi
 *
 *  The external electric field term is just f = rho E_0
 *
 *  We allow hydro to be NULL, in which case there is no force.
 *
 ****************************************************************************/

int psi_force_grad_mu(psi_t * psi, hydro_t * hydro) {

  int ic, jc, kc, index;
  int zs, ys, xs;
  int nlocal[3];

  double rho_elec;
  double f[3];
  double e0[3];

  double eunit, reunit, kt;

  if (hydro == NULL) return 0;
  assert(psi);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 1);

  physics_e0(e0);
  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;
  physics_kt(&kt);

  /* Memory strides */
  coords_strides(&xs, &ys, &zs);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	/* "Internal" field */

	f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs]);
	f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys]);
	f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs]);

	/* External field */

	f[X] += rho_elec*e0[X];
	f[Y] += rho_elec*e0[Y];
	f[Z] += rho_elec*e0[Z];

	f[X] *= reunit * kt;
	f[Y] *= reunit * kt;
	f[Z] *= reunit * kt;

	hydro_f_local_add(hydro, index, f);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_gradmu_conserve
 *
 *  This routine computes the force on the fluid via the gradient
 *  of the chemical potential.
 *
 *  First, we compute a correction which ensures global net momentum
 *  is unchanged. This must take account of colloids, if present.
 *  (There is no direct force on the colloid in this approach.)
 *
 *  This requires MPI_Allreduce().
 *
 *  The resultant force is accumulated to the hydrodynamic sector.
 *  If hydro is NULL, this routine is a bit over-the-top, but will
 *  compute the force on colloids.
 *
 *  The same formulation is used at the same time to compute the
 *  net force on the colloid. As this includes terms related to
 *  fluid properties near the edge of the particle, the force is
 *  not exactly qE. This is apparently significant, particularly
 *  at higher q, e.g., in getting the correct electrophoretic
 *  colloid speeds cf O'Brien and White. (Just using qE tends to
 *  give a higher forces and higher speeds.)
 *
 *  One is relying on overall electroneutrality for this to be a
 *  sensible procedure (ie., there should in principle be zero
 *  net force on the system).
 *
 *****************************************************************************/

int psi_force_gradmu_conserve(psi_t * psi, hydro_t * hydro,
		map_t * map, colloids_info_t * cinfo) {

  int ic, jc, kc;
  int nk;
  int nlocal[3];
  int index;

  double rho_elec;
  double f[3];
  double flocal[4] = {0.0, 0.0, 0.0, 0.0};
  double fsum[4];
  double e0[3], elocal[3];
  double eunit, reunit, kt;

  colloid_t * pc = NULL;
  MPI_Comm comm;

  assert(psi);
  assert(cinfo);

  physics_e0(e0); 

  coords_nlocal(nlocal);
  coords_cart_comm(psi->cs, &comm);

  psi_unit_charge(psi, &eunit);
  reunit = 1.0/eunit;
  physics_kt(&kt);

  psi_nk(psi, &nk);
  assert(nk == 2); /* This routine is not completely general */

  /* Compute force without correction. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	colloids_info_map(cinfo, index, &pc);

	psi_rho_elec(psi, index, &rho_elec);

	psi_electric_field_d3qx(psi, index, elocal);

	/* If solid, accumulate contribution to colloid;
	   otherwise to fluid node */

	f[X] = rho_elec*reunit*kt*(e0[X] + elocal[X]);
	f[Y] = rho_elec*reunit*kt*(e0[Y] + elocal[Y]);
	f[Z] = rho_elec*reunit*kt*(e0[Z] + elocal[Z]);

	if (pc) {

	  pc->force[X] += f[X];
	  pc->force[Y] += f[Y];
	  pc->force[Z] += f[Z];

	}
	else {

          /* Include ideal gas contribution */   
/*

	  physics_kt(&kt); 

	  for (n = 0; n < nk; n++) {

	    psi_grad_rho_d3qx(psi, map, index, n, grad_rho);

	    f[X] -= kt*grad_rho[X];
	    f[Y] -= kt*grad_rho[Y];
	    f[Z] -= kt*grad_rho[Z];
	  }
*/
	  if (hydro) hydro_f_local_add(hydro, index, f);

	  flocal[3] += 1.0;

	}

	/* Accumulate contribution to total force on system */

	flocal[X] += f[X];
	flocal[Y] += f[Y];
	flocal[Z] += f[Z];
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

        f[X] = - fsum[X];
        f[Y] = - fsum[Y];
        f[Z] = - fsum[Z];

	if (hydro) hydro_f_local_add(hydro, index, f);
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

int psi_force_divstress(psi_t * psi, hydro_t * hydro, colloids_info_t * cinfo) {

  int ic, jc, kc;
  int index, index1;
  int ia;
  int nlocal[3];

  double force[3];
  double pth1[3][3];

  colloid_t * pc = NULL;
  void (* chemical_stress)(const int index, double s[3][3]);

  assert(psi);
  assert(cinfo);

  coords_nlocal(nlocal);
  chemical_stress = fe_electro_stress_ex;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
 
       /* Calculate divergence based on 6-pt stencil */

	index1 = coords_index(ic+1, jc, kc);
	chemical_stress(index1, pth1);

	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X]);
	}

	index1 = coords_index(ic-1, jc, kc);
        chemical_stress(index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] += 0.5*(pth1[ia][X]);
        }

        index1 = coords_index(ic, jc+1, kc);
        chemical_stress(index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] -= 0.5*(pth1[ia][Y]);
        }

        index1 = coords_index(ic, jc-1, kc);
        chemical_stress(index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] += 0.5*(pth1[ia][Y]);
        }

        index1 = coords_index(ic, jc, kc+1);
        chemical_stress(index1, pth1);
        for (ia = 0; ia < 3; ia++) {
          force[ia] -= 0.5*(pth1[ia][Z]);
        }

        index1 = coords_index(ic, jc, kc-1);
        chemical_stress(index1, pth1);
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

int psi_force_divstress_d3qx(psi_t * psi, hydro_t * hydro, map_t * map,
			     colloids_info_t * cinfo) {

  int ic, jc, kc;
  int index, index_nb;
  int status, status_nb;
  int ia, ib;
  int nlocal[3];

  double force[3];
  double pth_nb[3][3];

  colloid_t * pc = NULL;
  void (* chemical_stress)(const int index, double s[3][3]);

  int p;
  int coords[3], coords_nb[3];

  assert(psi);
  assert(cinfo);

  coords_nlocal(nlocal);
  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	colloids_info_map(cinfo, index, &pc);

	coords_index_to_ijk(psi->cs, index, coords);

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

	  chemical_stress(index_nb, pth_nb);

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
 *  psi_force_divergence_set
 *
 *****************************************************************************/

int psi_force_divergence_set(const int flag) {
 
  psi_force_divergence_ = flag;

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_is_divergence
 *
 *****************************************************************************/

int psi_force_is_divergence(int * flag) {

  assert(flag);
  * flag = psi_force_divergence_;

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
 *****************************************************************************/

int psi_force_divstress_one_sided_d3qx(psi_t * psi, hydro_t * hydro,
				       map_t * map, colloids_info_t * cinfo) {

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

  colloid_t * pc = NULL;
  void (* chemical_stress)   (const int index, double s[3][3]);
  void (* chemical_stress_ex)(const int index, double s[3][3]);

  assert(psi);
  assert(cinfo);

  coords_nlocal(nlocal);
  chemical_stress    = fe_electro_stress;
  chemical_stress_ex = fe_electro_stress_ex;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	colloids_info_map(cinfo, index, &pc);

	coords_index_to_ijk(psi->cs, index, coords);

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

	    chemical_stress_ex(index_nb, pth_nb);

	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][ib] * psi_gr_cv[p][ib];
	      }
	    }

	  }

          if (status == MAP_FLUID && status_nb == MAP_FLUID) {

	    chemical_stress(index_nb, pth_nb);

	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][X] * psi_gr_cv[p][X];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][Y] * psi_gr_cv[p][Y];
	      force[ia] -= psi_gr_wv[p] * psi_gr_rcs2 * pth_nb[ia][Z] * psi_gr_cv[p][Z];
	    }
	  }

          if (status == MAP_FLUID && status_nb != MAP_FLUID) {

	    /* Current site r */
	    chemical_stress(index, pth);

	    /* Site r - cv */
	    coords1[X] = coords[X] - psi_gr_cv[p][X];
	    coords1[Y] = coords[Y] - psi_gr_cv[p][Y];
	    coords1[Z] = coords[Z] - psi_gr_cv[p][Z];

	    index1 = coords_index(coords1[X], coords1[Y], coords1[Z]);

	    chemical_stress(index1, pth1);

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

	    chemical_stress(index2, pth2);

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
