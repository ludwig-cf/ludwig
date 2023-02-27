/*****************************************************************************
 *
 *  subgrid.c
 *
 *  Routines for point-like particles.
 *
 *  See Nash et al. (2007).
 *
 *  Edinburgh Soft Matter and Statistical Phyiscs Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "colloid_sums.h"
#include "util.h"
#include "subgrid.h"
#include "field.h"

static double d_peskin(double);
static int subgrid_interpolation(colloids_info_t * cinfo, hydro_t * hydro);

static const double drange_ = 1.0; /* Max. range of interpolation - 1 */

/*****************************************************************************
 *
 *  subgrid_force_from_particles()
 *
 *  For each particle, accumulate the force on the relevant surrounding
 *  lattice nodes. Only nodes in the local domain are involved.
 *
 *****************************************************************************/


int subgrid_force_from_particles(colloids_info_t * cinfo, hydro_t * hydro,
				 wall_t * wall) {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nlocal[3], offset[3];
  int ncell[3];

  /* This is to get the correct expression of total_force */
  double drag, reta;
  double eta;

  PI_DOUBLE(pi);
  physics_t * phys = NULL;
  physics_ref(&phys);
  physics_eta_shear(phys, &eta);
  reta = 1.0/(6.0*pi*eta);
  /* ------------ */

  double r[3], r0[3], force[3], dcentre[3];
  double dr;
  colloid_t * p_colloid = NULL;  /* Subgrid colloid */
  colloid_t * presolved = NULL;  /* Resolved colloid occupuing node */

  assert(cinfo);
  if (hydro) { assert(hydro); }

  assert(wall);

  if (cinfo->nsubgrid == 0) return 0;

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  /* Add any wall lubrication corrections before communication to
   * find total external force on each particle */

  subgrid_wall_lubrication(cinfo, wall);
  colloid_sums_halo(cinfo, COLLOID_SUM_FORCE_EXT_ONLY);

  /* Loop through all cells (including the halo cells) */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

	 
          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;

  	  drag = reta*(1.0/p_colloid->s.ah - 1.0/p_colloid->s.al);

	  p_colloid->s.total_force[X] = drag*p_colloid->fex[X];
	  p_colloid->s.total_force[Y] = drag*p_colloid->fex[Y];
	  p_colloid->s.total_force[Z] = drag*p_colloid->fex[Z];
	  
	  /* Extract centre of the vesicle */
	  cs_minimum_distance(cinfo->cs, p_colloid->centerofmass, p_colloid->s.r, dcentre);

	  /* Accumulate total torque on vesicle (Sum of all the torque exerted by each bead with respect to the centre of vesicle) */
	  p_colloid->s.total_torque[X] += dcentre[Y]*p_colloid->s.total_force[Z]- dcentre[Z]*p_colloid->s.total_force[Y];
	  p_colloid->s.total_torque[Y] += dcentre[Z]*p_colloid->s.total_force[X]- dcentre[X]*p_colloid->s.total_force[Z];
	  p_colloid->s.total_torque[Z] += dcentre[X]*p_colloid->s.total_force[Y]- dcentre[Y]*p_colloid->s.total_force[X];
	  
/* -----> CHEMOVESICLE V2 */
/* Central particle does not interact with fluid */
          //if (p_colloid->s.iscentre == 1) continue; //LIGHTHOUSE
/* <----- */

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

	  i_min = imax(1,         (int) floor(r0[X] - drange_));
	  i_max = imin(nlocal[X], (int) ceil (r0[X] + drange_));
	  j_min = imax(1,         (int) floor(r0[Y] - drange_));
	  j_max = imin(nlocal[Y], (int) ceil (r0[Y] + drange_));
	  k_min = imax(1,         (int) floor(r0[Z] - drange_));
	  k_max = imin(nlocal[Z], (int) ceil (r0[Z] + drange_));

	  for (i = i_min; i <= i_max; i++) {
	    for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = cs_index(cinfo->cs, i, j, k);

		/* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - 1.0*i;
		r[Y] = r0[Y] - 1.0*j;
		r[Z] = r0[Z] - 1.0*k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);
            
		force[X] = p_colloid->fex[X]*dr;
		force[Y] = p_colloid->fex[Y]*dr;
		force[Z] = p_colloid->fex[Z]*dr;

		colloids_info_map(cinfo, index, &presolved);

		if (presolved == NULL) {
		  if (hydro) {
		    hydro_f_local_add(hydro, index, force);
		  }
		}
		else {
		  double rd[3] = {};
		  double torque[3] = {};
		  presolved->force[X] += force[X];
		  presolved->force[Y] += force[Y];
		  presolved->force[Z] += force[Z];
		  rd[X] = 1.0*i - (presolved->s.r[X] - 1.0*offset[X]);
		  rd[Y] = 1.0*j - (presolved->s.r[Y] - 1.0*offset[Y]);
		  rd[Z] = 1.0*k - (presolved->s.r[Z] - 1.0*offset[Z]);
		  cross_product(rd, force, torque);
		  presolved->torque[X] += torque[X];
		  presolved->torque[Y] += torque[Y];
		  presolved->torque[Z] += torque[Z];
                }

	      }
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  subgrid_update
 *
 *  This function is responsible for update of position for
 *  sub-gridscale particles. It takes the place of BBL for
 *  fully resolved particles.
 *
 *****************************************************************************/

int subgrid_update(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  double drag, reta;
  double eta;
  PI_DOUBLE(pi);
  colloid_t * p_colloid;
  physics_t * phys = NULL;

  double ran[2];    /* Random numbers for fluctuation dissipation correction */
  double frand[3];  /* Random force */
  double kt;        /* Temperature */

  assert(cinfo);
  if (hydro) assert(hydro);

  if (cinfo->nsubgrid == 0) return 0;

  colloids_info_ncell(cinfo, ncell);

  if (hydro) subgrid_interpolation(cinfo, hydro);

  colloid_sums_halo(cinfo, COLLOID_SUM_SUBGRID);

  /* Loop through all cells (including the halo cells) */

  physics_ref(&phys);
  physics_kt(phys, &kt);
  physics_eta_shear(phys, &eta);
  reta = 1.0/(6.0*pi*eta);

  /* Loop through all edge subgrid particles */ 

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;
  
  	  drag = reta*(1.0/p_colloid->s.ah - 1.0/p_colloid->s.al);
  
  	  if (noise_flag == 0) {
  	    frand[X] = 0.0; frand[Y] = 0.0; frand[Z] = 0.0;
  	  }
  	  else {
  	    for (ia = 0; ia < 3; ia++) {
  	      while (1) {
  	          /* To keep the random correction smaller than 3 sigma.
  	           * Otherwise, a large thermal fluctuation may cause a
  	           * numerical problem. */
  	        util_ranlcg_reap_gaussian(&p_colloid->s.rng, ran);
  	        if (fabs(ran[0]) < 3.0) {
  	          frand[ia] = sqrt(2.0*kt*drag)*ran[0];
  	          break;
  	        }
  	        if (fabs(ran[1]) < 3.0) {
  	          frand[ia] = sqrt(2.0*kt*drag)*ran[1];
  	          break;
  	        }
              }
            }
          }
  
          for (ia = 0; ia < 3; ia++) {
	    if (hydro) {
	      p_colloid->s.v[ia] = p_colloid->fsub[ia] + drag*p_colloid->fex[ia] + frand[ia];
	    }
	    else {
	      p_colloid->s.v[ia] = drag*p_colloid->fex[ia] + frand[ia];
	    }

  	    p_colloid->s.dr[ia] = p_colloid->s.v[ia];
  	  }
	}
      }
    }
  }
  return 0;
}


/*****************************************************************************
 *
*  subgrid_centre_update  ! MUST BE CALLED AFTER SUBGRID_UPDATE !
 *
 *  This function is responsible for update of position for
 *  sub-gridscale particles. It takes the place of BBL for
 *  fully resolved particles.
 *
 *****************************************************************************/

int subgrid_centre_update(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag) {

  
  int ia;
  int rank;
  int ic, jc, kc, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];
  int nparticles, ntotal[3];
  int timestep, freq = 100;
  double centerofmass[3] = {0. , 0. , 0.}, dr[3] = {0., 0., 0.};
  double r_edge[3];

  MPI_Comm comm;
  MPI_Status status;
  cs_cart_comm(cinfo->cs, &comm);


  colloid_t * p_colloid;
  colloid_t * p_colloid_edge;
  physics_t * phys = NULL;
  physics_ref(&phys);
  timestep = physics_control_timestep(phys);
  assert(cinfo);
  assert(hydro);

  if (cinfo->nsubgrid == 0) return 0;

  colloid_sums_halo(cinfo, COLLOID_SUM_SUBGRID);
  colloids_info_ncell(cinfo, ncell);
  cs_ntotal(cinfo->cs, ntotal);

  /* <-------------------------------Loop over local domain to find central--------------------------------> */

  for (ic = 1; ic <= ncell[X]; ic++) {
    colloids_info_climits(cinfo, X, ic, di);
    for (jc = 1; jc <= ncell[Y]; jc++) {
      colloids_info_climits(cinfo, Y , jc, dj);
      for (kc = 1; kc <= ncell[Z]; kc++) {
        colloids_info_climits(cinfo, Z, kc, dk);

        colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);
	for (; p_colloid; p_colloid = p_colloid->next) {
	  /* Filter out non subgrid and edge particles */	
          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;
          if (p_colloid->s.iscentre != 1) continue;
   	    
	  for (ia = 0; ia < 3; ia++) centerofmass[ia] = 0.0;
	  for (ia = 0; ia < 3; ia++) r_edge[ia] = 0.0;
	  nparticles = 0;
    	  /*<--------------------------Loop over central-adjacent cells to find edge particles----------------------------> */
    	  for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
    	    for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
    	      for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
      
                colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &p_colloid_edge);

    	        for (; p_colloid_edge; p_colloid_edge = p_colloid_edge->next) {
    	          /* Filter out particles from vesicles not belonging to the current p_colloid */
    	          if ((p_colloid_edge->s.indexcentre == p_colloid->s.index) && (p_colloid_edge->s.iscentre == 0)) {
		    nparticles += 1;
		    cs_minimum_distance(cinfo->cs, p_colloid->s.r, p_colloid_edge->s.r, r_edge);
    	            for (ia = 0; ia < 3; ia ++) {
		      centerofmass[ia] += p_colloid->s.r[ia] + r_edge[ia];
		    }
    	          }
    	        }
    	      }
    	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    centerofmass[ia] = centerofmass[ia] /nparticles;
	    p_colloid->centerofmass[ia] = centerofmass[ia];

            p_colloid->s.v[ia] = centerofmass[ia] - p_colloid->s.r[ia]; 
	    dr[ia] = centerofmass[ia] - p_colloid->s.r[ia];

            p_colloid->s.dr[ia] = p_colloid->s.v[ia];
	  }
	  if (timestep % freq == 0){
	  //pe_verbose(cinfo->pe, "centerofmass = %14.7e\n", centerofmass[Z]);
	  //pe_verbose(cinfo->pe, "position z = %14.7e\n", p_colloid->s.r[Z]);
	  //pe_verbose(cinfo->pe, "dr z = %14.7e\n", dr[Z]);
	  }
	}
      }
    }
  }

/* Communicate the dr to all processes */
  int count, myid, realrank = 1500;

  MPI_Barrier(comm);
  MPI_Comm_size(comm, &count);
  MPI_Comm_rank(comm, &myid);

  if (centerofmass[X] + centerofmass[Y] + centerofmass[Z] != 0.0) {
    realrank = myid;
  }

  if (myid == realrank) {
    // If we are the process that found the real central particle, send centerofmass to everyone
    int i;
    for (i = 0; i < count; i++) {
      if (i != realrank) {
        MPI_Send(dr, 3, MPI_DOUBLE, i, 0, comm);
      }
    }
  } 

  if (myid != realrank) {
    // If we are a receiver process, receive centerofmass
    MPI_Recv(dr, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
  }

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
		
        colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);
	for (; p_colloid; p_colloid = p_colloid->next) {
          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;
          if (p_colloid->s.iscentre != 1) continue;

	  for (ia = 0; ia < 3; ia ++) p_colloid->s.dr[ia] = dr[ia];
	}
      }
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  subgrid_interpolation
 *
 *  Interpolate (delta function method) the lattice velocity field
 *  to the position of the particles.
 *
 *****************************************************************************/

static int subgrid_interpolation(colloids_info_t * cinfo, hydro_t * hydro) {

  int ic, jc, kc;
  int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max;
  int index;
  int nlocal[3], offset[3];
  int ncell[3];

  double r0[3], r[3], u[3];
  double dr;
  colloid_t * p_colloid;

  assert(cinfo);
  assert(hydro);

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  /* Loop through all cells (including the halo cells) and set
   * the velocity at each particle to zero for this step. */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;

/* -----> CHEMOVESICLE V2 */ //LIGHTHOUSE
	  //if (p_colloid->s.iscentre == 1) continue;
/* <----- */
	  p_colloid->fsub[X] = 0.0;
	  p_colloid->fsub[Y] = 0.0;
	  p_colloid->fsub[Z] = 0.0;

	  p_colloid->s.fsub[X] = 0.0;
	  p_colloid->s.fsub[Y] = 0.0;
	  p_colloid->s.fsub[Z] = 0.0;
	
	}
      }
    }
  }

  /* And add up the contributions to the velocity from the lattice. */

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_colloid);

	for ( ; p_colloid; p_colloid = p_colloid->next) {

          if (p_colloid->s.type != COLLOID_TYPE_SUBGRID) continue;

/* -----> CHEMOVESICLE V2 */
	  //if (p_colloid->s.iscentre == 1) continue;
/* <----- */

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  /* Work out which local lattice sites are involved
	   * and loop around */

	  i_min = imax(1,         (int) floor(r0[X] - drange_));
	  i_max = imin(nlocal[X], (int) ceil (r0[X] + drange_));
	  j_min = imax(1,         (int) floor(r0[Y] - drange_));
	  j_max = imin(nlocal[Y], (int) ceil (r0[Y] + drange_));
	  k_min = imax(1,         (int) floor(r0[Z] - drange_));
	  k_max = imin(nlocal[Z], (int) ceil (r0[Z] + drange_));

	  for (i = i_min; i <= i_max; i++) {
	    for (j = j_min; j <= j_max; j++) {
	      for (k = k_min; k <= k_max; k++) {

		index = cs_index(cinfo->cs, i, j, k);

		/* Separation between r0 and the coordinate position of
		 * this site */

		r[X] = r0[X] - 1.0*i;
		r[Y] = r0[Y] - 1.0*j;
		r[Z] = r0[Z] - 1.0*k;

		dr = d_peskin(r[X])*d_peskin(r[Y])*d_peskin(r[Z]);
		hydro_u(hydro, index, u);

		p_colloid->fsub[X] += u[X]*dr;
		p_colloid->fsub[Y] += u[Y]*dr;
		p_colloid->fsub[Z] += u[Z]*dr;

		p_colloid->s.fsub[X] += u[X]*dr;
		p_colloid->s.fsub[Y] += u[Y]*dr;
		p_colloid->s.fsub[Z] += u[Z]*dr;

	      }
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  subgrid_wall_lubrication
 *
 *  Accumulate lubrication corrections to the external force on each particle.
 *
 *****************************************************************************/

int subgrid_wall_lubrication(colloids_info_t * cinfo, wall_t * wall) {

  double drag[3];
  colloid_t * pc = NULL;

  double f[3] = {0.0, 0.0, 0.0};
  
  assert(cinfo);
  assert(wall);

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.type != COLLOID_TYPE_SUBGRID) continue;
    wall_lubr_sphere(wall, pc->s.ah, pc->s.r, drag);
    pc->fex[X] += drag[X]*pc->s.v[X];
    pc->fex[Y] += drag[Y]*pc->s.v[Y];
    pc->fex[Z] += drag[Z]*pc->s.v[Z];
    f[X] -= drag[X]*pc->s.v[X];
    f[Y] -= drag[Y]*pc->s.v[Y];
    f[Z] -= drag[Z]*pc->s.v[Z];
  }

  wall_momentum_add(wall, f);

  return 0;
}

/*****************************************************************************
 *
 *  d_peskin
 *
 *  Approximation to \delta(r) according to Peskin.
 *
 *****************************************************************************/

static double d_peskin(double r) {

  double rmod;
  double delta = 0.0;

  rmod = fabs(r);

  if (rmod <= 1.0) {
    delta = 0.125*(3.0 - 2.0*rmod + sqrt(1.0 + 4.0*rmod - 4.0*rmod*rmod));
  }
  else if (rmod <= 2.0) {
    delta = 0.125*(5.0 - 2.0*rmod - sqrt(-7.0 + 12.0*rmod  - 4.0*rmod*rmod));
  }

  return delta;
}



/*****************************************************************************
 *
 *  subgrid_flux_mask  ! MUST BE CALLED AFTER SUBGRID_UPDATE !
 *  TODO: fuse finding m and n and rcentre as well as their broadcasting ?
 *  TODO: realrank is a variable that can be set when calculating m 
 *
 *****************************************************************************/

int subgrid_flux_mask(pe_t * pe, colloids_info_t * cinfo, field_t * flux_mask, field_t * u_mask, rt_t * rt, field_t * phi, map_t * map, hydro_t * hydro) {

  int ia, i, j, k, index, p, template_on, beads_per_vesicle;
  int nlocal[3], offset[3];
  int my_id, id, totnum_ids, hole_id = -1, ortho_id = -1, centre_id = -1;
  int centrefound = 0, holefound = 0, orthofound = 0;
  int mask_phi_switch, mask_psi_switch, vesicle_number, reaction_switch, interaction_mask;
  int timestep, writefreq;
  int correction = 0;

  double m[3] = {0., 0., 0.}, n[3] = {0., 0., 0.}, ijk[3];
  double r[3], rcentre[3], rhole[3], rortho[3], rcentre_local[3], rsq;
  double rnorm, norm_m, norm_n;
  double cosalpha, alpha, gaussalpha, std_alpha, alpha_cutoff, std_width;
  double radius, gaussr;  
  double permeability[2];
  double kappa, kappa1, kappam1;
  double phicopy;
  double totphi_local = 0.0, totpsi_local = 0.0, totphi, totpsi;

  char reaction_model[BUFSIZ];
  char template[BUFSIZ];

  FILE * fp;

  typedef struct {
    int vesicle_index[2];
    double rcentre[3*2];
    double rhole[3*2];
  } vTuple; 

  colloid_t * pc;
  physics_t * phys = NULL;

  MPI_Comm comm;
  MPI_Status status;
  cs_cart_comm(cinfo->cs, &comm);


  physics_ref(&phys);
  physics_rvesicle(phys, &radius);

  timestep = physics_control_timestep(phys);

  cs_nlocal_offset(cinfo->cs, offset);
  cs_nlocal(cinfo->cs, nlocal);

  MPI_Comm_rank(comm, &my_id);
  MPI_Comm_size(comm, &totnum_ids);
  
  assert(cinfo);


  /* Loop through all cells (including the halo cells) and set
   * the mask at each node to 1 */
  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cinfo->cs, i, j, k);
	flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 0)] = 1.0;
	flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 1)] = 1.0;
	u_mask->data[addr_rank0(flux_mask->nsites, index)] = 0.0;
      }
    }
  }

  rt_int_parameter(rt, "freq_write", &writefreq);
  rt_int_parameter(rt, "vesicle_number", &vesicle_number);
  template_on = rt_string_parameter(rt, "vesicle_template", template, BUFSIZ);

  if (template_on == 0) pe_fatal(pe,"You have to specify vesicle_template\n");
  if (strcmp(template, "icosphere") == 0) {
    beads_per_vesicle = 43;
  }
  if (strcmp(template, "fullerene") == 0) {
    beads_per_vesicle = 61;
  }
  if (strcmp(template, "hexasphere") == 0) {
    beads_per_vesicle = 241;
  }
  if (strcmp(template, "trisphere") == 0) {
    beads_per_vesicle = 643;
  }
  
  interaction_mask = rt_switch(rt, "phi_interaction_mask");  

  if (vesicle_number != 0) {

    mask_phi_switch = rt_switch(rt, "mask_phi_switch");
    mask_psi_switch = rt_switch(rt, "mask_psi_switch");
    reaction_switch = rt_switch(rt, "chemical_reaction_switch");

    /* Go in anyway to get the txt files ??? LIGHTHOUSE */ 
    /*if (!mask_phi_switch && !mask_psi_switch && !reaction_switch && !interaction_mask) return 0; */

    if (interaction_mask || mask_phi_switch) { 
      p = rt_double_parameter(rt, "mask_phi_permeability", &permeability[0]);
      if (p != 1) pe_fatal(pe, "Please specify mask_phi_permeability\n");
    }
    if (interaction_mask || mask_psi_switch) {
      p = rt_double_parameter(rt, "mask_psi_permeability", &permeability[1]);
      if (p != 1) pe_fatal(pe, "Please specify mask_psi_permeability\n");
    }
    if (interaction_mask || mask_phi_switch || mask_psi_switch) {
      p = rt_double_parameter(rt, "mask_std_alpha", &std_alpha);
      if (p != 1) pe_fatal(pe, "Please specify mask_std_alpha\n");
      p = rt_double_parameter(rt, "mask_std_width", &std_width);
      if (p != 1) pe_fatal(pe, "Please specify std_width\n");
      p = rt_double_parameter(rt, "mask_alpha_cutoff", &alpha_cutoff);
      if (p != 1) pe_fatal(pe, "Please specify mask_alpha_cutoff\n");
    }

    if (reaction_switch) {
      p = rt_string_parameter(rt, "chemical_reaction_model", reaction_model, BUFSIZ );
      if (p != 1) pe_fatal(pe, "Please specify chemical_reaction_model\n");
      else {
        if (strcmp(reaction_model, "uniform") == 0) {
	  p = rt_double_parameter(rt, "kappa", &kappa);
	  if (p != 1) pe_fatal(pe, "If you choose chemical_reaction_model uniform, please specify kappa\n");
	}
	else if (strcmp(reaction_model, "psi<->phi") == 0) {
	  p = rt_double_parameter(rt, "kappa1", &kappa1);
	  if (p != 1) pe_fatal(pe, "If you choose chemical_reaction_model psi<->phi, please specify kappa1\n");
	  p = rt_double_parameter(rt, "kappam1", &kappam1);
	  if (p != 1) pe_fatal(pe, "If you choose chemical_reaction_model psi<->phi, please specify kappam1\n");
        } 

	else pe_fatal(pe, "Key chemical_reaction_model can only take one of the following values: uniform, psi<->phi\n");
      } 
    }
  }
  else if (reaction_switch) pe_fatal(cinfo->pe, "Reaction not available without vesicle\n");
  else return 0;

/* The following is only useful when a vesicle is present */

  /* Find central particle */
  colloids_info_local_head(cinfo, &pc);
  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.indexcentre != 1) continue;
    if (pc->s.iscentre == 1) {
      centrefound = 1;
      rcentre[0] = pc->s.r[0];
      rcentre[1] = pc->s.r[1];
      rcentre[2] = pc->s.r[2];
      MPI_Comm_rank(comm, &centre_id);
    }

    if (pc->s.ishole == 1) {
      holefound = 1;
      rhole[0] = pc->s.r[0];
      rhole[1] = pc->s.r[1];
      rhole[2] = pc->s.r[2];
      MPI_Comm_rank(comm, &hole_id);
    }

/*    if (pc->s.index == 164) {
      orthofound = 1;
      rortho[0] = pc->s.r[0]; 
      rortho[1] = pc->s.r[1]; 
      rortho[2] = pc->s.r[2]; 
      MPI_Comm_rank(comm, &ortho_id);
    }
*/
  }

  MPI_Barrier(comm);

/* If you found it, send it. Else, receive it */
  if (centrefound == 1) {
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rcentre, 3, MPI_DOUBLE, id, 0, comm);
    }
  } 
  MPI_Barrier(comm);
  if (centrefound != 1) {
    MPI_Recv(rcentre, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(comm);

  if (holefound == 1) {
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rhole, 3, MPI_DOUBLE, id, 0, comm);
    }
  }
  MPI_Barrier(comm);
  if (holefound != 1) {
    MPI_Recv(rhole, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
  }


  MPI_Barrier(comm);

/*
  if (orthofound == 1) {
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rortho, 3, MPI_DOUBLE, id, 0, comm);
    }
  }
  MPI_Barrier(comm);
  if (orthofound != 1) {
    MPI_Recv(rortho, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(comm);
*/

  cs_minimum_distance(cinfo->cs, rcentre, rhole, m);
//  cs_minimum_distance(cinfo->cs, rcentre, rortho, n);
    
  norm_m = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
//  norm_n = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);

  for (int ia = 0; ia < 3; ia++) {
    m[ia] /= norm_m;
//    n[ia] /= norm_n;
  }

  colloids_info_all_head(cinfo, &pc);
  for ( ; pc; pc = pc->nextall) {
//    if (pc->s.indexcentre != 1) continue; LIGHTHOUSE when using several vesicles
    for (int ia = 0; ia < 3; ia++) {
      pc->s.m[ia] = m[ia];
      pc->centerofmass[ia] = rcentre[ia];
//      pc->s.n[ia] = n[ia];
    }
  }
 

/* Assign mask value */

  rcentre_local[X] = rcentre[X] - offset[X];
  rcentre_local[Y] = rcentre[Y] - offset[Y];
  rcentre_local[Z] = rcentre[Z] - offset[Z];

  for (i = 1; i <= nlocal[X] + 0; i++) {
    for (j = 1; j <= nlocal[Y] + 0; j++) {
      for (k = 1; k <= nlocal[Z] + 0; k++) {

        index = cs_index(cinfo->cs, i, j, k); 
	
        ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

        cs_minimum_distance(cinfo->cs, rcentre_local, ijk , r);

	rsq = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
	rnorm = sqrt(rsq);

	// PRODUCE PHI
        if (reaction_switch) {
	  if (rnorm < radius - std_width - 2) {
	    if (strcmp(reaction_model, "uniform") == 0) 
	    {
	      phi->data[addr_rank1(phi->nsites, 2, index, 0)] += kappa;
	    }
	    else if (strcmp(reaction_model, "psi<->phi") == 0)
	    {
	      phicopy = phi->data[addr_rank1(phi->nsites, 2, index, 0)];
	      phi->data[addr_rank1(phi->nsites, 2, index, 0)] += kappa1*phi->data[addr_rank1(phi->nsites, 2, index, 1)] - kappam1*phi->data[addr_rank1(phi->nsites, 2, index, 0)];
	      phi->data[addr_rank1(phi->nsites, 2, index, 1)] += -kappa1*phi->data[addr_rank1(phi->nsites, 2, index, 1)] + kappam1*phicopy;
	    }
	  }
	}

	totphi_local += phi->data[addr_rank1(phi->nsites, 2, index, 0)];
	totpsi_local += phi->data[addr_rank1(phi->nsites, 2, index, 1)];

	// ASSIGN MASK VALUE

	// To avoid dividing by 0
	// Allowed because the central particle should never be near the edge of the vesicle
	if (rnorm == 0.0) continue;

	if (interaction_mask || mask_phi_switch) { 

          cosalpha = (r[X]*m[X] + r[Y]*m[Y] + r[Z]*m[Z])/rnorm;

	  if (cosalpha > 1.) cosalpha = 1.; 
	  if (cosalpha < -1.) cosalpha = -1.; 

          alpha = acos(cosalpha);

	  if (std_alpha == 0.0) {
	    gaussalpha = 0.0;
	  }
	  else {
            gaussalpha = exp(-.5*(alpha/std_alpha)*(alpha/std_alpha)*(alpha/std_alpha)*(alpha/std_alpha));
	  }

          gaussr = exp(-.5*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width));
	    
/*
	  if (alpha*alpha > alpha_cutoff*alpha_cutoff) {
	    flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 0)] = 1. - (1. - permeability[0])*gaussr*(1 - gaussalpha);
	  }
*/
	  flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 0)] = 1. - (1. - permeability[0])*gaussr*(1 - gaussalpha);
	  u_mask->data[addr_rank1(u_mask->nsites, 1, index, 0)] = gaussalpha;
	}

	if (interaction_mask || mask_psi_switch) { 

          cosalpha = (r[X]*m[X] + r[Y]*m[Y] + r[Z]*m[Z]) / rnorm;
 
	  if (cosalpha > 1.) cosalpha = 1.; 
	  if (cosalpha < -1.) cosalpha = -1.; 

          alpha = acos(cosalpha);

	  if (std_alpha == 0.0) {
	    gaussalpha = 0.0;
	  }
	  else {
            gaussalpha = exp(-.5*(alpha/std_alpha)*(alpha/std_alpha)*(alpha/std_alpha)*(alpha/std_alpha));
	  }

          gaussr = exp(-.5*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width)*(rnorm - radius)/(std_width));
 
          /*if (alpha*alpha > alpha_cutoff*alpha_cutoff) {
	    flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 1)] = 1. - (1. - permeability[0])*gaussr*(1 - gaussalpha);
	  }
	*/
	  flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 1)] = 1. - (1. - permeability[0])*gaussr*(1 - gaussalpha);
	  u_mask->data[addr_rank1(u_mask->nsites, 1, index, 0)] = gaussalpha;
        }

	assert(flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 0)] >= 0.0);
	assert(flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 0)] <= 1.0);
	assert(flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 1)] >= 0.0);
	assert(flux_mask->data[addr_rank1(flux_mask->nsites, 2, index, 1)] <= 1.0);

      }
    }
  }

  // These subroutines require communication of total quantities of phi and/or psi
  rt_int_parameter(rt, "psi_grad_mu_correction", &correction);
  if ((strcmp(reaction_model, "psi<->phi") == 0) && correction) {
    MPI_Allreduce(&totpsi_local, &totpsi, 1, MPI_DOUBLE, MPI_SUM, comm);
    //pe_verbose(cinfo->pe, "totpsi = %f\n", totpsi);
    int is_grad_mu_psi = 0; 
    int nsfluid;
    double rvolume;
    double f[3];
    double mu_psi[3] = {};
    double3 grad_mu_psi = {};

    {
      physics_grad_mu_psi(phys, mu_psi);

      grad_mu_psi.x = mu_psi[X];
      grad_mu_psi.y = mu_psi[Y];
      grad_mu_psi.z = mu_psi[Z];
 
      is_grad_mu_psi = (mu_psi[X] != 0.0 || mu_psi[Y] != 0.0 || mu_psi[Z] != 0.0);
    }

    if (is_grad_mu_psi) {
      assert(map);
      map_volume_allreduce(map, MAP_FLUID, &nsfluid);
      rvolume = 1.0/nsfluid;
      for (ia = 0; ia < 3; ia++) {
	// Force on fluid is - psi grad_mu_ext
	f[ia] = rvolume*totpsi*mu_psi[ia];
      }
    }

    for (i = 1; i <= nlocal[X] + 0; i++) {
      for (j = 1; j <= nlocal[Y] + 0; j++) {
        for (k = 1; k <= nlocal[Z] + 0; k++) {
          index = cs_index(cinfo->cs, i, j, k); 
	  hydro_f_local_add(hydro, index, f);
	}
      }
    }
  }

  if (timestep % writefreq == 0) {
    MPI_Reduce(&totphi_local, &totphi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&totpsi_local, &totpsi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (my_id == 0) {
      fp = fopen("totphi.txt", "a");
      fprintf(fp, "%14.7e,       %14.7e\n", totphi, totpsi);
      fclose(fp);
    }
  }
  return 0;
}


/*****************************************************************************
 * *  subgrid_flux_mask_vesicle2  ! MUST BE CALLED AFTER SUBGRID_UPDATE !
 *  TODO: fuse finding m and n and rcentre as well as their broadcasting ?
 *  TODO: realrank is a variable that can be set when calculating m 
 *
 *****************************************************************************/

int subgrid_flux_mask_vesicle2(colloids_info_t * cinfo, field_t * flux_mask, rt_t * rt) {

  int ia, i, j, k, on, index;
  int nlocal[3], offset[3];
  int my_id, id, totnum_ids, hole_id = -1, ortho_id = -1, centre_id = -1;
  int centrefound = 0, holefound = 0, orthofound = 0;

  double m[3] = {0., 0., 0.}, n[3] = {0., 0., 0.}, ijk[3];
  double r[3], rcentre[3], rhole[3], rortho[3], rcentre_local[3], rsq;
  double rnorm, norm_m, norm_n, mobility;
  double cosalpha, alpha, gaussalpha;
  double rvesicle;  

  MPI_Comm comm;
  MPI_Status status;
  cs_cart_comm(cinfo->cs, &comm);

  colloid_t * pc;
  physics_t * phys = NULL;

  physics_ref(&phys);
  physics_mobility(phys, &mobility);
  physics_rvesicle(phys, &rvesicle);

  cs_nlocal_offset(cinfo->cs, offset);
  cs_nlocal(cinfo->cs, nlocal);

  MPI_Comm_rank(comm, &my_id);
  MPI_Comm_size(comm, &totnum_ids);
  
  if (cinfo->nsubgrid == 0) return 0;
  assert(cinfo);

  rt_int_parameter(rt, "subgrid_flux_mask", &on); 
  
  /* Loop through all cells (including the halo cells) and set
   * the mobility at each node to default mobiltiy for this step. */
/*
  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cinfo->cs, i, j, k);
	field_scalar_set(flux_mask, index, mobility);
      }
    }
  }
*/
  if (!on) return 0;


/* Find central particle */

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.indexcentre != 242) continue;
    if (pc->s.iscentre == 1) {
      centrefound = 1;
      rcentre[0] = pc->s.r[0];
      rcentre[1] = pc->s.r[1];
      rcentre[2] = pc->s.r[2];
      MPI_Comm_rank(comm, &centre_id);
    }

    if (pc->s.ishole == 1) {
      holefound = 1;
      rhole[0] = pc->s.r[0];
      rhole[1] = pc->s.r[1];
      rhole[2] = pc->s.r[2];
      MPI_Comm_rank(comm, &hole_id);
    }
/*
    if (pc->s.index == 164) {
      orthofound = 1;
      rortho[0] = pc->s.r[0]; 
      rortho[1] = pc->s.r[1]; 
      rortho[2] = pc->s.r[2]; 
      MPI_Comm_rank(comm, &ortho_id);
    }
*/
  }
  //pe_verbose(cinfo->pe, "I am process %d centrefound = %d, holefound = %d, orthofound = %d\n", my_id, centrefound, holefound, orthofound);

  MPI_Barrier(comm);

  if (centrefound == 1) {
    //pe_verbose(cinfo->pe, "sending rcentre = %f %f %f\n", rcentre[0], rcentre[1], rcentre[2]);
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rcentre, 3, MPI_DOUBLE, id, 0, comm);
    }
  } 
  MPI_Barrier(comm);
  if (centrefound != 1) {
    MPI_Recv(rcentre, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
    //pe_verbose(cinfo->pe, "receiving rcentre = %f %f %f\n", rcentre[0], rcentre[1], rcentre[2]);
  }

  MPI_Barrier(comm);

  if (holefound == 1) {
    //pe_verbose(cinfo->pe, "sending rhole = %f %f %f\n", rhole[0], rhole[1], rhole[2]);
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rhole, 3, MPI_DOUBLE, id, 0, comm);
    }
  }
  MPI_Barrier(comm);
  if (holefound != 1) {
    MPI_Recv(rhole, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
    //pe_verbose(cinfo->pe, "receiving rhole = %f %f %f\n", rhole[0], rhole[1], rhole[2]);
  }

/*
  MPI_Barrier(comm);
  if (orthofound == 1) {
    for (int id = 0; id < totnum_ids; id++) {
      if (my_id == id) continue;
      MPI_Send(rortho, 3, MPI_DOUBLE, id, 0, comm);
    }
  }
  MPI_Barrier(comm);
  if (orthofound != 1) {
    MPI_Recv(rortho, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
  }
*/
  MPI_Barrier(comm);
  //pe_verbose(cinfo->pe, "rcentre = %f %f %f rhole = %f %f %f\n", rcentre[X], rcentre[Y], rcentre[Z], rhole[X], rhole[Y], rhole[Z]);

  cs_minimum_distance(cinfo->cs, rcentre, rhole, m);
  cs_minimum_distance(cinfo->cs, rcentre, rortho, n);
    
  norm_m = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
  norm_n = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  //pe_verbose(cinfo->pe, "n = %f %f %f\n", n[X], n[Y], n[Z]);
  //pe_verbose(cinfo->pe, "m = %f %f %f\n", m[X], m[Y], m[Z]);

  for (int ia = 0; ia < 3; ia++) {
    m[ia] /= norm_m;
    n[ia] /= norm_n;
  }

  //pe_verbose(cinfo->pe, "rortho = %f %f %f, m = %f %f %f n = %f %f %f\n", my_id, rortho[X], rortho[Y], rortho[Z], m[0], m[1], m[2], n[0], n[1], n[2]);

  colloids_info_all_head(cinfo, &pc);
  for ( ; pc; pc = pc->nextall) {
    if (pc->s.indexcentre != 242) continue;
    for (int ia = 0; ia < 3; ia++) {
      pc->s.m[ia] = m[ia];
      pc->s.n[ia] = n[ia];
      pc->centerofmass[ia] = rcentre[ia];
    }
  }
 
// Assign mobility

  rcentre_local[X] = rcentre[X] - offset[X];
  rcentre_local[Y] = rcentre[Y] - offset[Y];
  rcentre_local[Z] = rcentre[Z] - offset[Z];
 
  //pe_verbose(cinfo->pe,"rcentre local = %f %f %f\n", rcentre_local[X], rcentre_local[Y], rcentre_local[Z]);

  for (i = 1; i <= nlocal[X] + 0; i++) {
    for (j = 1; j <= nlocal[Y] + 0; j++) {
      for (k = 1; k <= nlocal[Z] + 0; k++) {

        index = cs_index(cinfo->cs, i, j, k); 
	
        ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

        cs_minimum_distance(cinfo->cs, rcentre_local, ijk , r);

	rsq = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
	rnorm = sqrt(rsq);

	if (rnorm <= rvesicle + 1 && rnorm >= rvesicle - 1) {
	  field_scalar_set(flux_mask, index, 0.0);
          cosalpha = (r[X]*m[X] + r[Y]*m[Y] + r[Z]*m[Z]) / rnorm;
          alpha = acos(cosalpha);

          //if (alpha >= -1.0 && alpha <= 1.0) {
          //  gaussalpha = exp(-0.5*(alpha/0.6)*(alpha/0.6))*mobility;
	  //  field_scalar_set(flux_mask, index, gaussalpha);
	  //}
          if (alpha >= -0.5 && alpha <= 0.5) {
            field_scalar_set(flux_mask, index, mobility);
          }
	}
      }
    }
  }
  return 0;
}


