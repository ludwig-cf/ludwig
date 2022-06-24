/*****************************************************************************
 *
 *  interact.c
 *
 *  Colloid interactions with external fields (single particle);
 *  and colloid-colloid interactions.
 *
 *  The idea here is to provide call-backs for each type of interaction
 *     lubrication, pairwise, bonds, angles
 *  which is registered depending on the exact one required. However,
 *  this code does not care about the details.
 *
 *  Each interaction present may give rise to a potential and some
 *  statistics on separation, cutoffs etc.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "control.h"
#include "stats_colloid.h"
#include "driven_colloid.h"
#include "interaction.h"

struct interact_s {
  pe_t * pe;
  cs_t * cs;
  double vlocal[INTERACT_MAX];       /* Local potential contributions */
  double vtotal[INTERACT_MAX];       /* Total potential contributions */

  int    rcset[INTERACT_MAX];        /* Centre-centre interaction active */
  double rc[INTERACT_MAX];           /* Centre-centre cutoff range */

  int    hcset[INTERACT_MAX];        /* Surface-surface interaction active */
  double hc[INTERACT_MAX];           /* Surface-surface cutoff range */

  void * abstr[INTERACT_MAX];        /* Abstract interaction types */
  compute_ft compute[INTERACT_MAX];  /* Corresponding compute functions */
  stat_ft stats[INTERACT_MAX];       /* Statisitics functions */
};

/*****************************************************************************
 *
 *  interact_create
 *
 *****************************************************************************/

int interact_create(pe_t * pe, cs_t * cs, interact_t ** pobj) {

  interact_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (interact_t *) calloc(1, sizeof(interact_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(interact_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  interact_free
 *
 *****************************************************************************/

void interact_free(interact_t * obj) {

  assert(obj);

  free(obj);

  return;
}

/*****************************************************************************
 *
 *  interact_rc_set
 *
 *****************************************************************************/

int interact_rc_set(interact_t * obj, interact_enum_t it, double rc) {

  assert(obj);
  assert(it < INTERACT_MAX);

  obj->rcset[it] = 1;
  obj->rc[it] = rc;

  return 0;
}

/*****************************************************************************
 *
 *  interact_hc_set
 *
 *****************************************************************************/

int interact_hc_set(interact_t * obj, interact_enum_t it, double hc) {

  assert(obj);
  assert(it < INTERACT_MAX);

  obj->hcset[it] = 1;
  obj->hc[it] = hc;

  return 0;
}

/*****************************************************************************
 *
 *  interact_potential_add
 *
 *****************************************************************************/

int interact_potential_add(interact_t * obj, interact_enum_t it,
			   void * potential, compute_ft compute) {

  assert(obj);
  assert(it < INTERACT_MAX);
  assert(potential);
  assert(compute);

  obj->abstr[it] = potential;
  obj->compute[it] = compute;

  return 0;
}

/*****************************************************************************
 *
 *  interact_statistic_add
 *
 *****************************************************************************/

int interact_statistic_add(interact_t * obj, interact_enum_t it, void * pot,
			   stat_ft stats) {

  assert(obj);
  assert(it < INTERACT_MAX);
  assert(pot);
  assert(stats);

  obj->abstr[it] = pot;
  obj->stats[it] = stats;

  return 0;
}

/* -----> CHEMOVESICLE V2 */
/* Now calculates PHI<->SUBGRID interaction only when phi_subgrid_on input key is on
*/

/*****************************************************************************
 *
 *  interact_compute
 *
 *  Top-level function for compuatation of external forces to be called
 *  once per time step. Note that particle copies in the halo regions
 *  must have zero external force/torque on exit from this routine.
 *
 *****************************************************************************/

int interact_compute(interact_t * interact, colloids_info_t * cinfo,
		     map_t * map, psi_t * psi, ewald_t * ewald, field_t * phi,
			field_t * subgrid_potential, rt_t * rt) {

  int nc;
  int on;

  assert(interact);
  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);

  if (nc > 0) {
    colloids_update_forces_zero(cinfo);
    colloids_update_forces_external(cinfo, psi);
    colloids_update_forces_fluid_gravity(cinfo, map);
    colloids_update_forces_fluid_driven(cinfo, map);
    interact_wall(interact, cinfo);

    rt_int_parameter(rt, "phi_subgrid_switch", &on);
    if (on) colloids_update_discrete_forces_phi(cinfo, phi, subgrid_potential); 

    if (nc > 1) {
      interact_bonds(interact, cinfo);
      interact_bonds_harmonic(interact, cinfo);
      interact_bonds_harmonic2(interact, cinfo);
      interact_bonds_harmonic3(interact, cinfo);

      interact_pairwise(interact, cinfo);
      interact_angles(interact, cinfo);
      interact_angles_harmonic(interact, cinfo);
      interact_angles_dihedral(interact, cinfo);
      if (ewald) ewald_sum(ewald);
    }

    if (is_statistics_step()) {

      pe_info(interact->pe, "\nParticle statistics:\n");

      interact_stats(interact, cinfo);
      pe_info(interact->pe, "\n");
      stats_colloid_velocity_minmax(cinfo);
    }
    
    colloids_update_forces_ext(cinfo);
  }

  return 0;
}

/*****************************************************************************
 *
 *  interact_stats
 *
 *****************************************************************************/

int interact_stats(interact_t * obj, colloids_info_t * cinfo) {

  int nc = 0;
  void * intr = NULL;
  double stats[INTERACT_STAT_MAX];
  double hminlocal, hmin;
  double rminlocal, rmin;
  double rmaxlocal, rmax;
  double vlocal, v;
  MPI_Comm comm;

  colloids_info_ntotal(cinfo, &nc);
  pe_mpi_comm(obj->pe, &comm);
  
  if (nc > 0) {

    intr = obj->abstr[INTERACT_WALL];
    
    if (intr) {
      
      obj->stats[INTERACT_WALL](intr, stats);
      
      hminlocal = stats[INTERACT_STAT_HMINLOCAL];
      vlocal = stats[INTERACT_STAT_VLOCAL];

      MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
      
      pe_info(obj->pe, "Wall potential minimum h is: %14.7e\n", hmin);
      pe_info(obj->pe, "Wall potential energy is:    %14.7e\n", v);
    }

    if (nc > 1) {
  
      intr = obj->abstr[INTERACT_LUBR];

     if (intr) {

	obj->stats[INTERACT_LUBR](intr, stats);

	hminlocal = stats[INTERACT_STAT_HMINLOCAL];
	MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	pe_info(obj->pe, "Lubrication minimum h is:    %14.7e\n", hmin);
      }

      intr = obj->abstr[INTERACT_PAIR];

      if (intr) {

	obj->stats[INTERACT_PAIR](intr, stats);

	hminlocal = stats[INTERACT_STAT_HMINLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Pair potential minimum h is: %14.7e\n", hmin);
	pe_info(obj->pe, "Pair potential energy is:    %14.7e\n", v);
      }

      intr = obj->abstr[INTERACT_BOND];

      if (intr) {

	obj->stats[INTERACT_BOND](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Bond potential minimum r is: %14.7e\n", rmin);
	pe_info(obj->pe, "Bond potential maximum r is: %14.7e\n", rmax);
	pe_info(obj->pe, "Bond potential energy is:    %14.7e\n", v);
      }

      //CHANGE1
      intr = obj->abstr[INTERACT_BOND_HARMONIC];

      if (intr) {

	obj->stats[INTERACT_BOND_HARMONIC](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Bond harmonic potential minimum r is: %14.7e\n", rmin);
	pe_info(obj->pe, "Bond harmonic potential maximum r is: %14.7e\n", rmax);
	pe_info(obj->pe, "Bond harmonic potential energy is:    %14.7e\n", v);
	
      }

      intr = obj->abstr[INTERACT_BOND_HARMONIC2];

      if (intr) {

	obj->stats[INTERACT_BOND_HARMONIC2](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Bond harmonic2 potential minimum r is: %14.7e\n", rmin);
	pe_info(obj->pe, "Bond harmonic2 potential maximum r is: %14.7e\n", rmax);
	pe_info(obj->pe, "Bond harmonic2 potential energy is:    %14.7e\n", v);
	

      }

      intr = obj->abstr[INTERACT_BOND_HARMONIC3];

      if (intr) {

	obj->stats[INTERACT_BOND_HARMONIC3](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Bond harmonic3 potential minimum r is: %14.7e\n", rmin);
	pe_info(obj->pe, "Bond harmonic3 potential maximum r is: %14.7e\n", rmax);
	pe_info(obj->pe, "Bond harmonic3 potential energy is:    %14.7e\n", v);

      }

      intr = obj->abstr[INTERACT_ANGLE];

      if (intr) {

	obj->stats[INTERACT_ANGLE](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Angle minimum angle is:      %14.7e\n", rmin);
	pe_info(obj->pe, "Angle maximum angle is:      %14.7e\n", rmax);
	pe_info(obj->pe, "Angle potential energy is:   %14.7e\n", v);
      }

      //CHANGE1
      intr = obj->abstr[INTERACT_ANGLE_HARMONIC];

      //CHANGE1
      if (intr) {

	obj->stats[INTERACT_ANGLE_HARMONIC](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Angle harmonic minimum angle is:      %14.7e\n", rmin);
	pe_info(obj->pe, "Angle harmonic maximum angle is:      %14.7e\n", rmax);
	pe_info(obj->pe, "Angle harmonic potential energy is:   %14.7e\n", v);
      }

      //CHANGE1
      intr = obj->abstr[INTERACT_ANGLE_DIHEDRAL];

      //CHANGE1
      if (intr) {

	obj->stats[INTERACT_ANGLE_DIHEDRAL](intr, stats);

	rminlocal = stats[INTERACT_STAT_RMINLOCAL];
	rmaxlocal = stats[INTERACT_STAT_RMAXLOCAL];
	vlocal = stats[INTERACT_STAT_VLOCAL];

	MPI_Reduce(&rminlocal, &rmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
	MPI_Reduce(&rmaxlocal, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&vlocal, &v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	pe_info(obj->pe, "Angle dihedral minimum angle is:      %14.7e\n", rmin);
	pe_info(obj->pe, "Angle dihedral maximum angle is:      %14.7e\n", rmax);
	pe_info(obj->pe, "Angle dihedral potential energy is:   %14.7e\n", v);
      }
    }
  }
  
  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_zero_set
 *
 *  Set the external forces on the particles to zero (including halos).
 *  All additional forces are then accumulated.
 *
 *****************************************************************************/

int colloids_update_forces_zero(colloids_info_t * cinfo) {

  colloid_t * pc = NULL;

  assert(cinfo);

  colloids_info_all_head(cinfo, &pc);

  for (; pc; pc = pc->nextall) {
    pc->force[X] = 0.0;
    pc->force[Y] = 0.0;
    pc->force[Z] = 0.0;
    pc->torque[X] = 0.0;
    pc->torque[Y] = 0.0;
    pc->torque[Z] = 0.0;

/*  The following is for extraction purposes */
    pc->s.fphi[X] = 0.0;
    pc->s.fphi[Y] = 0.0;
    pc->s.fphi[Z] = 0.0;

    pc->s.fsprings[X] = 0.0;
    pc->s.fsprings[Y] = 0.0;
    pc->s.fsprings[Z] = 0.0;

  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_single_particle_set
 *
 *  Accumulate single particle force contributions.
 *
 *  psi may be NULL, in which case, assume no charged species, otherwise
 *  we assume two. Indeed, not used at the moment.
 *
 *****************************************************************************/

int colloids_update_forces_external(colloids_info_t * cinfo, psi_t * psi) {

  int ic, jc, kc, ia;
  int ncell[3];
  double b0[3];          /* external fields */
  double g[3];
  double btorque[3];
  double dforce[3];
  colloid_t * pc;
  physics_t * phys = NULL;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  physics_ref(&phys);
  physics_b0(phys, b0);
  physics_fgrav(phys, g);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	for (; pc; pc = pc->next) {

	  /* All particles have gravity */
	  pc->force[X] += g[X];
	  pc->force[Y] += g[Y];
	  pc->force[Z] += g[Z];

          if (pc->s.type == COLLOID_TYPE_SUBGRID) continue;

	  btorque[X] = pc->s.s[Y]*b0[Z] - pc->s.s[Z]*b0[Y];
	  btorque[Y] = pc->s.s[Z]*b0[X] - pc->s.s[X]*b0[Z];
	  btorque[Z] = pc->s.s[X]*b0[Y] - pc->s.s[Y]*b0[X];

	  driven_colloid_force(pc->s.s, dforce);

	  for (ia = 0; ia < 3; ia++) {
	    pc->torque[ia] += btorque[ia];         /* Magnetic field */
	    pc->force[ia] += dforce[ia];           /* Active force */
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_forces_fluid_gravity_set
 *
 *  Set the current gravtitational force on the fluid. This should
 *  match, exactly, the force on the colloids and so depends on the
 *  current number of fluid sites globally (fluid volume).
 *
 *  Note the volume calculation involves a collective communication.
 *
 *****************************************************************************/

int colloids_update_forces_fluid_gravity(colloids_info_t * cinfo,
					 map_t * map) {
  int nc;
  int ia;
  int nsfluid;
  int is_gravity = 0;
  double rvolume;
  double g[3], f[3];
  physics_t * phys = NULL;

  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);
  if (nc == 0) return 0;

  physics_ref(&phys);
  physics_fgrav(phys, g);
  is_gravity = (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0);

  if (is_gravity) {

    assert(map);
    map_volume_allreduce(map, MAP_FLUID, &nsfluid);
    rvolume = 1.0/nsfluid;

    /* Force per fluid node to balance is... */

    for (ia = 0; ia < 3; ia++) {
      f[ia] = -g[ia]*rvolume*nc;
    }

    physics_fbody_set(phys, f);
  }

  return 0;
}


/*****************************************************************************
 *
 *  colloid_forces_fluid_driven
 *
 *  Set the current drive force on the fluid. This should
 *  match, exactly, the force on the colloids and so depends on the
 *  current number of fluid sites globally (fluid volume).
 *
 *  Note the calculation involves a collective communication.
 *
 *  TODO: sort out wall momentum account fw. This is the reason
 *  for the commented-out code and assert(0).
 *
 *****************************************************************************/

int colloids_update_forces_fluid_driven(colloids_info_t * cinfo,
                                         map_t * map) {

  int nc;
  int ia;
  int nsfluid;
  double rvolume;
  int periodic[3];
  double fd[3], f[3];
  /* double fw[3]; */
  physics_t * phys = NULL;

  return 0; /* This routine needs an MOT before use. */
  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);

  if (nc == 0) return 0;

  physics_ref(&phys);

  if (is_driven()) {

    assert(map);
    map_volume_allreduce(map, MAP_FLUID, &nsfluid);
    rvolume = 1.0/nsfluid;

    /* Force per fluid node to balance is... */
    driven_colloid_total_force(cinfo, fd);
    
    for (ia = 0; ia < 3; ia++) {
      f[ia] = -1.0*fd[ia]*rvolume*periodic[ia];
      /* fw[ia] = -1.0*fd[ia]*(1.0 - periodic[ia])/(1.0*pe_size());*/
    }

    physics_fbody_set(phys, f);

    /* Need to account for wall momentum transfer */
    assert(0); /* NO TEST */
  }

  return 0;
}



/*****************************************************************************
 *
 *  colloids_update_discrete_forces_phi_old
 *  Old and wrong as it uses the sum of all potentials
 *
 *****************************************************************************/

int colloids_update_discrete_forces_phi_old(colloids_info_t * cinfo, field_t * phi, field_t * subgrid_potential) {

  int nlocal[3], offset[3], ncell[3];
  int i, j, k, ic, jc, kc;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int index, indexm1, indexp1, ia;

  double u0, delta, cutoff, phi_, rnorm, rsq, u;
  double r0[3], r[3], force[3], grad_u[3], um1, up1;
  colloid_t * pc = NULL;

/* -----> For book-keeping */
  int freq = 100000, timestep;
  double colloidforce[3], localforce[3], globalforce[3];
  FILE * fp;
// time stuff
  physics_t * phys = NULL;
  physics_ref(&phys);
  timestep = physics_control_timestep(phys);
// global communication stuff */
  MPI_Comm comm;
  cs_cart_comm(cinfo->cs, &comm);
/* <----- */

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  assert(cinfo);
  assert(phi);
  assert(phi->nf == 2);
  assert(subgrid_potential);

/* Should I also go through the halo cells ? No because you only calculate the values for the domain lattice */

  /* Initialize subgrid flux to 0 before loop over colloids */
  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
        index = cs_index(cinfo->cs, i, j, k);
        field_scalar_set(subgrid_potential, index, 0);
      }
    }
  }
  
  /* Loop through all cells (including the halo cells) */

  localforce[X] = 0.0;
  localforce[Y] = 0.0;
  localforce[Z] = 0.0;


  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
        colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
	
        /* Loop through all colloids */
        for (; pc; pc = pc->next) {

          u0 = pc->s.u0;
          delta = pc->s.delta;
          cutoff = pc->s.cutoff;

          /* Need to translate the colloid position to "local"
          * coordinates, so that the correct range of lattice
          * nodes is found */

          r0[X] = pc->s.r[X] - 1.0*offset[X];
          r0[Y] = pc->s.r[Y] - 1.0*offset[Y];
          r0[Z] = pc->s.r[Z] - 1.0*offset[Z];

          /* Work out which local lattice sites are involved
          * and loop around */

          i_min = imax(1,         (int) floor(r0[X] - cutoff + 1));
          i_max = imin(nlocal[X], (int) ceil (r0[X] + cutoff + 1));
          j_min = imax(1,         (int) floor(r0[Y] - cutoff + 1));
          j_max = imin(nlocal[Y], (int) ceil (r0[Y] + cutoff + 1));
          k_min = imax(1,         (int) floor(r0[Z] - cutoff + 1));
          k_max = imin(nlocal[Z], (int) ceil (r0[Z] + cutoff + 1));

          /* Initialisation of the force counter on colloid to 0 before the loop */

          colloidforce[X] = 0.0;
          colloidforce[Y] = 0.0;
          colloidforce[Z] = 0.0;

          for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
              for (k = k_min; k <= k_max; k++) {
		
                index = cs_index(cinfo->cs, i, j, k);

                /* Compute the distance between the colloid and the node  */

                r[X] = - r0[X] + 1.0*i;
                r[Y] = - r0[Y] + 1.0*j;
                r[Z] = - r0[Z] + 1.0*k;

                rsq = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
                rnorm = sqrt(rsq);

		if (rnorm > cutoff) continue;
					
                /* Retrieve phi */
                phi_ = phi->data[addr_rank1(phi->nsites, 2, index, 0)]; //LIGHTHOUSE

                /* Compute potential and grad of potential (grad_u) */
                u = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsq)/delta);
                subgrid_potential->data[addr_rank0(subgrid_potential->nsites, index)] += u;
}
            }
          }
        //Next colloid
        }
      }
    }
  }


  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
        colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
	
        /* Loop through all colloids */
        for (; pc; pc = pc->next) {

          u0 = pc->s.u0;
          delta = pc->s.delta;
          cutoff = pc->s.cutoff;

          /* Need to translate the colloid position to "local"
          * coordinates, so that the correct range of lattice
          * nodes is found */

          r0[X] = pc->s.r[X] - 1.0*offset[X];
          r0[Y] = pc->s.r[Y] - 1.0*offset[Y];
          r0[Z] = pc->s.r[Z] - 1.0*offset[Z];

          /* Work out which local lattice sites are involved
          * and loop around */

          i_min = imax(1,         (int) floor(r0[X] - cutoff - 1));
          i_max = imin(nlocal[X], (int) ceil (r0[X] + cutoff + 1));
          j_min = imax(1,         (int) floor(r0[Y] - cutoff - 1));
          j_max = imin(nlocal[Y], (int) ceil (r0[Y] + cutoff + 1));
          k_min = imax(1,         (int) floor(r0[Z] - cutoff - 1));
          k_max = imin(nlocal[Z], (int) ceil (r0[Z] + cutoff + 1));

          /* Initialisation of the force counter on colloid to 0 before the loop */

          colloidforce[X] = 0.0;
          colloidforce[Y] = 0.0;
          colloidforce[Z] = 0.0;

          for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
              for (k = k_min; k <= k_max; k++) {
		
                index = cs_index(cinfo->cs, i, j, k);

                /* Compute the distance between the colloid and the node  */

                r[X] = - r0[X] + 1.0*i;
                r[Y] = - r0[Y] + 1.0*j;
                r[Z] = - r0[Z] + 1.0*k;

                rsq = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
                rnorm = sqrt(rsq);

		if (rnorm > cutoff + 2) continue;
					
                /* Retrieve phi */
                phi_ = phi->data[addr_rank1(phi->nsites, 2, index, 0)];

                /* Compute potential and grad of potential (grad_u) */
                /* u = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsq)/delta);
                subgrid_potential->data[addr_rank0(subgrid_potential->nsites, index)] += u;
} */
 

		/* Compute gradient of u */
		/* x direction */
		{
		  indexm1 = cs_index(cinfo->cs, i-1, j, k);
		  indexp1 = cs_index(cinfo->cs, i+1, j, k);
		  field_scalar(subgrid_potential, indexm1, &um1);
		  field_scalar(subgrid_potential, indexp1, &up1);
		  grad_u[X] = 0.5*(up1 - um1);
		}
	
		/* y direction */
		{
		  indexm1 = cs_index(cinfo->cs, i, j-1, k);
		  indexp1 = cs_index(cinfo->cs, i, j+1, k);
		  field_scalar(subgrid_potential, indexm1, &um1);
		  field_scalar(subgrid_potential, indexp1, &up1);
		  grad_u[Y] = 0.5*(up1 - um1);
		}

	
		/* z direction */
		{
		  indexm1 = cs_index(cinfo->cs, i, j, k-1);
		  indexp1 = cs_index(cinfo->cs, i, j, k+1);
		  field_scalar(subgrid_potential, indexm1, &um1);
		  field_scalar(subgrid_potential, indexp1, &up1);
		  grad_u[Z] = 0.5*(up1 - um1);
		}

                /* Compute force  */
		
                force[X] = phi_*grad_u[X];
                force[Y] = phi_*grad_u[Y];
                force[Z] = phi_*grad_u[Z];

                pc->force[X] += force[X];
                pc->force[Y] += force[Y];
                pc->force[Z] += force[Z];

		colloidforce[X] += force[X];
		colloidforce[Y] += force[Y];
		colloidforce[Z] += force[Z];

                /* Add reaction of the colloid force to the node */
              }
            }
          }
	
	/* Increment the local force with the force on this colloid */
	//printf("colloid index %d, force %f %f %f \n", pc->s.index, colloidforce[X], colloidforce[Y], colloidforce[Z]);
	for (ia = 0; ia < 3; ia ++) localforce[ia] += colloidforce[ia];
        //printf("localforce = %f %f %f\n", localforce[X], localforce[Y], localforce[Z]);
        //Next colloid
        }
      }
    }
  }

/* For force book-keeping */
  if (timestep % freq == 0) {
    MPI_Allreduce(localforce, globalforce, 3, MPI_DOUBLE, MPI_SUM, comm);
    fp = fopen("force_colloid.txt", "a");
    fprintf(fp, "%14.7e %14.7e %14.7e\n", globalforce[X], globalforce[Y], globalforce[Z]);
    fclose(fp);
  }
  //Next cell list
  return 0;
}


/*****************************************************************************
 *
 *  colloids_update_discrete_forces_phi
 * 
 *  Here I go over the nodes within the interaction range of each subgrid particles 
 *  (+ 2 nodes for the centered difference of the gradient) to calculate the total force
 *  PHI exerts on it. 
 *
 *  I then sum up all the interaction potentials between node and
 *  subgid particles for later use in cahn_hilliard and phi_force
 *
 *  Possibility to write the total force exerted by PHI on all the subgrid particles
 *  in a file by at a frequency writefreq
 * 
 *****************************************************************************/

int colloids_update_discrete_forces_phi(colloids_info_t * cinfo, field_t * phi, field_t * subgrid_potential) {

  int nlocal[3], offset[3], ncell[3];
  int i, j, k, ic, jc, kc;
  int i_min, i_max, j_min, j_max, k_min, k_max;
  int index, ia;

  double phi_, u, um1, up1, u0, delta, cutoff;
  double rnorm, rsq, rsqm1, rsqp1;
  double r0[3], r[3], grad_u[3];

  colloid_t * pc = NULL;
  physics_t * phys = NULL;

/* ------------------------- For book-keeping ------------------------------> */
  int writefreq = 100000, timestep;
  double colloidforce[3], localforce[3] = {0., 0., 0.}, globalforce[3];
  FILE * fp;

// time stuff
  physics_ref(&phys);
  timestep = physics_control_timestep(phys);

// global communication stuff
  MPI_Comm comm;
  cs_cart_comm(cinfo->cs, &comm);
/* <------------------------------------------------------------------------- */

  cs_nlocal(cinfo->cs, nlocal);
  cs_nlocal_offset(cinfo->cs, offset);
  colloids_info_ncell(cinfo, ncell);

  assert(cinfo);
  assert(phi);
  assert(phi->nf == 2);
  assert(subgrid_potential);

  /* Initialize subgrid potential to 0 */
  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
        index = cs_index(cinfo->cs, i, j, k);
        field_scalar_set(subgrid_potential, index, 0);
      }
    }
  }
  
  /* Go over cell lists */
  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {
        colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
	
        /* Go over subgrid particles */
        for (; pc; pc = pc->next) {

          u0 = pc->s.u0;
          delta = pc->s.delta;
          cutoff = pc->s.cutoff;

          /* Need to translate the colloid position to "local"
          * coordinates, so that the correct range of lattice
          * nodes is found */

          r0[X] = pc->s.r[X] - 1.0*offset[X];
          r0[Y] = pc->s.r[Y] - 1.0*offset[Y];
          r0[Z] = pc->s.r[Z] - 1.0*offset[Z];

          /* Work out which local lattice sites are involved
          * and loop around */

          i_min = imax(1,         (int) floor(r0[X] - cutoff - 1));
          i_max = imin(nlocal[X], (int) ceil (r0[X] + cutoff + 1));
          j_min = imax(1,         (int) floor(r0[Y] - cutoff - 1));
          j_max = imin(nlocal[Y], (int) ceil (r0[Y] + cutoff + 1));
          k_min = imax(1,         (int) floor(r0[Z] - cutoff - 1));
          k_max = imin(nlocal[Z], (int) ceil (r0[Z] + cutoff + 1));

          /* Initialisation of the force counter on subgrid 
	  *  particles to 0 before the loop */

          colloidforce[X] = 0.0;
          colloidforce[Y] = 0.0;
          colloidforce[Z] = 0.0;

          for (i = i_min; i <= i_max; i++) {
            for (j = j_min; j <= j_max; j++) {
              for (k = k_min; k <= k_max; k++) {
		
	        /* For each node in the interaction range of the subgrid particle:
                *    i)  calculate the force PHI GRAD(U) it exerts on the particle
	        *    ii) increment the total interaction potential in field_t * 
	        *	 subgrid_potential for later use by cahn_hilliard and phi_force*/

                index = cs_index(cinfo->cs, i, j, k);
                phi_ = phi->data[addr_rank1(phi->nsites, 2, index, 0)];

		/* Compute gradient of u using centered difference */
		/* x direction */
		{
                  r[Y] = - r0[Y] + 1.0*j;
                  r[Z] = - r0[Z] + 1.0*k;

                  r[X] = - r0[X] + 1.0*(i-1);
		  rsqm1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

                  r[X] = - r0[X] + 1.0*(i+1);
		  rsqp1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

		  if (rsqm1 > cutoff*cutoff) um1 = 0.0;
		  else um1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqm1)/delta);
		  if (rsqp1 > cutoff*cutoff) up1 = 0.0;
		  else up1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqp1)/delta);
		  
		  grad_u[X] = 0.5*(up1 - um1);
		}
	
		/* y direction */
		{
                  r[X] = - r0[X] + 1.0*i;
                  r[Z] = - r0[Z] + 1.0*k;

                  r[Y] = - r0[Y] + 1.0*(j-1);
		  rsqm1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

                  r[Y] = - r0[Y] + 1.0*(j+1);
		  rsqp1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

		  if (rsqm1 > cutoff*cutoff) um1 = 0.0;
		  else um1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqm1)/delta);
		  if (rsqp1 > cutoff*cutoff) up1 = 0.0;
		  else up1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqp1)/delta);

		  grad_u[Y] = 0.5*(up1 - um1);
		}
	
		/* z direction */
		{
                  r[X] = - r0[X] + 1.0*i;
                  r[Y] = - r0[Y] + 1.0*j;

                  r[Z] = - r0[Z] + 1.0*(k-1);
		  rsqm1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

                  r[Z] = - r0[Z] + 1.0*(k+1);
		  rsqp1 = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];

		  if (rsqm1 > cutoff*cutoff) um1 = 0.0;
		  else um1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqm1)/delta);
		  if (rsqp1 > cutoff*cutoff) up1 = 0.0;
		  else up1 = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsqp1)/delta);

		  grad_u[Z] = 0.5*(up1 - um1);
		}
	
                /* Calculate force */
		
                pc->force[X] += phi_*grad_u[X];
                pc->force[Y] += phi_*grad_u[Y];
                pc->force[Z] += phi_*grad_u[Z];

		
		pc->s.fphi[X] += phi_*grad_u[X];
                pc->s.fphi[Y] += phi_*grad_u[Y];
                pc->s.fphi[Z] += phi_*grad_u[Z];


		colloidforce[X] += phi_*grad_u[X];
		colloidforce[Y] += phi_*grad_u[Y];
		colloidforce[Z] += phi_*grad_u[Z];

  		//if (timestep % writefreq == 0 && grad_u[Y] != 0.0) printf("interaction %14.7e\n", colloidforce[Y]);

		/* Increment the total interaction potential for later use by cahn_hilliard and phi_force */

                r[X] = - r0[X] + 1.0*i;
                r[Y] = - r0[Y] + 1.0*j;
                r[Z] = - r0[Z] + 1.0*k;

                rsq = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
                rnorm = sqrt(rsq);

		if (rnorm > cutoff) continue;

                u = u0/(sqrt(2*M_PI*delta))*exp(-0.5*(rsq)/delta);
                subgrid_potential->data[addr_rank0(subgrid_potential->nsites, index)] += u;
              }
            }
          }
	
	/* Increment the local force with the force on this colloid */
	//printf("colloid index %d, force %f %f %f \n", pc->s.index, colloidforce[X], colloidforce[Y], colloidforce[Z]);
	//if (pc->s.index == 1) pe_verbose(cinfo->pe, "%14.7e %14.7e %14.7e \n", colloidforce[X], colloidforce[Y], colloidforce[Z]);
	for (ia = 0; ia < 3; ia ++) localforce[ia] += colloidforce[ia];
        //pe_verbose(cinfo->pe, "local phi force = %f %f %f\n", localforce[X], localforce[Y], localforce[Z]);
        //pe_verbose(cinfo->pe, "local force = %f %f %f\n", pc->force[X], pc->force[Y], pc->force[Z]);

        //Next colloid
        }
      }
    }
  }

/* Output sum of force exerted on the subgrid particles */
  if (timestep % writefreq == 0) {
    MPI_Allreduce(localforce, globalforce, 3, MPI_DOUBLE, MPI_SUM, comm);
    fp = fopen("TOT_INTERACT_FORCE_SUBGRID.txt", "a");
    fprintf(fp, "%14.7e %14.7e %14.7e\n", globalforce[X], globalforce[Y], globalforce[Z]);
    fclose(fp);
  }
  //Next cell list
  return 0;
}


/*****************************************************************************
 *
 *  interact_pairwise
 *
 *****************************************************************************/

int interact_pairwise(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_LUBR];
  if (intr) obj->compute[INTERACT_LUBR](cinfo, intr);

  intr = obj->abstr[INTERACT_PAIR];
  if (intr) obj->compute[INTERACT_PAIR](cinfo, intr);

  return 0;
}

/*****************************************************************************
 *
 *  interact_wall
 *
 *****************************************************************************/

int interact_wall(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_WALL];
  if (intr) obj->compute[INTERACT_WALL](cinfo, intr);

  return 0;
}

/*****************************************************************************
 *
 *  interact_bonds
 *
 *****************************************************************************/

int interact_bonds(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_BOND];
  if (intr) interact_find_bonds_all(obj, cinfo, 0);
  if (intr) obj->compute[INTERACT_BOND](cinfo, intr);

  return 0;
}

//CHANGE1
/*****************************************************************************
 *
 *  interact_bonds_harmonic
 *
 *****************************************************************************/

int interact_bonds_harmonic(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_BOND_HARMONIC];
  if (intr) interact_find_bonds_all(obj, cinfo, 1);
  if (intr) obj->compute[INTERACT_BOND_HARMONIC](cinfo, intr);

  return 0;
}

/*****************************************************************************
 *
 *  interact_bonds_harmonic2
 *
 *****************************************************************************/

int interact_bonds_harmonic2(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_BOND_HARMONIC2];
  if (intr) interact_find_bonds_all2(obj, cinfo, 1);
  if (intr) obj->compute[INTERACT_BOND_HARMONIC2](cinfo, intr);

  return 0;
}


/*****************************************************************************
 *
 *  interact_bonds_harmonic3
 *
 *****************************************************************************/

int interact_bonds_harmonic3(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_BOND_HARMONIC3];
  if (intr) interact_find_bonds_all3(obj, cinfo, 1);
  if (intr) obj->compute[INTERACT_BOND_HARMONIC3](cinfo, intr);

  return 0;
}

/*****************************************************************************
 *
 *  interact_angles
 *
 *****************************************************************************/

int interact_angles(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_ANGLE];
  if (intr) obj->compute[INTERACT_ANGLE](cinfo, intr);

  return 0;
}

//CHANGE1
/*****************************************************************************
 *
 *  interact_angles_harmonic
 *
 *****************************************************************************/
int interact_angles_harmonic(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_ANGLE_HARMONIC];
  if (intr) obj->compute[INTERACT_ANGLE_HARMONIC](cinfo, intr);

  return 0;
}

//CHANGE1
/*****************************************************************************
 *
 *  interact_angles_dihedral
 *
 *****************************************************************************/
int interact_angles_dihedral(interact_t * obj, colloids_info_t * cinfo) {

  void * intr = NULL;

  assert(obj);
  assert(cinfo);

  intr = obj->abstr[INTERACT_ANGLE_DIHEDRAL];
  if (intr) obj->compute[INTERACT_ANGLE_DIHEDRAL](cinfo, intr);

  return 0;
}

/*****************************************************************************
 *
 *  interact_find_bonds
 *
 *  For backwards compatability, the case where only local bonds are
 *  included (nextra = 0).
 *
 *****************************************************************************/

int interact_find_bonds(interact_t * obj, colloids_info_t * cinfo) {

  assert(obj);
  assert(cinfo);

  interact_find_bonds_all(obj, cinfo, 1);

  return 0;
}


/*****************************************************************************
 *
 *  interact_find_bonds_all
 *
 *  Examine the local colloids and match any bonded interactions
 *  in terms of pointers.
 *
 *  Include nextra cells in each direction into the halo region.
 *
 *****************************************************************************/

int interact_find_bonds_all(interact_t * obj, colloids_info_t * cinfo,
			    int nextra) {

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int n1, n2;
  int ncell[3];

  int nbondfound = 0;
  int nbondpair = 0;
 
  colloid_t * pc1;
  colloid_t * pc2;

  assert(obj);
  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1 - nextra; ic1 <= ncell[X] + nextra; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1 - nextra; jc1 <= ncell[Y] + nextra; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1 - nextra; kc1 <= ncell[Z] + nextra; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {
	  //printf("%d %d %d %d \n", pc1->s.index, pc1->s.nbonds, pc1->s.nbonds2, pc1->s.nbonds3);

	  if (pc1->s.nbonds == 0) continue;

	  for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
	    for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
	      for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
   
		colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &pc2);
		for (; pc2; pc2 = pc2->next) {

		  if (pc2->s.nbonds == 0) continue; 
                  
		  for (n1 = 0; n1 < pc1->s.nbonds; n1++) {
		    if (pc1->s.bond[n1] == pc2->s.index) {
		      nbondfound += 1;
		      pc1->bonded[n1] = pc2;
		     
		      /* And bond is reciprocated */
		      for (n2 = 0; n2 < pc2->s.nbonds; n2++) {
			if (pc2->s.bond[n2] == pc1->s.index) {
			  nbondpair += 1;
			  pc2->bonded[n2] = pc1;
			}
		      }
		    }
		  }

		  /* Cell list */
	        }
	      }
	    }
	  }
	}
      }
    }
  }
  if (nbondfound != nbondpair) {
    /* There is a mismatch in the bond information (treat as fatal) */
    pe_fatal(obj->pe, "Find bonds: bond not reciprocated\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  interact_find_bonds2
 *
 *  For backwards compatability, the case where only local bonds are
 *  included (nextra = 0).
 *
 *****************************************************************************/

int interact_find_bonds2(interact_t * obj, colloids_info_t * cinfo) {

  assert(obj);
  assert(cinfo);

  interact_find_bonds_all2(obj, cinfo, 1);

  return 0;
}

/*****************************************************************************
 *
 *  interact_find_bonds_all2
 *
 *  Examine the local colloids and match any bonded interactions
 *  in terms of pointers.
 *
 *  Include nextra cells in each direction into the halo region.
 *
 *****************************************************************************/

int interact_find_bonds_all2(interact_t * obj, colloids_info_t * cinfo,
			    int nextra) {

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int n1, n2;
  int ncell[3];

  int nbondfound = 0;
  int nbondpair = 0;

  colloid_t * pc1;
  colloid_t * pc2;

  assert(obj);
  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1 - nextra; ic1 <= ncell[X] + nextra; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1 - nextra; jc1 <= ncell[Y] + nextra; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1 - nextra; kc1 <= ncell[Z] + nextra; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {
	  if (pc1->s.nbonds2 == 0) continue;

	  for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
	    for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
	      for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
   
		colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &pc2);
		for (; pc2; pc2 = pc2->next) {

		  if (pc2->s.nbonds2 == 0) continue; 
                  
		  for (n1 = 0; n1 < pc1->s.nbonds2; n1++) {
		    if (pc1->s.bond2[n1] == pc2->s.index) {
		      nbondfound += 1;
		      pc1->bonded2[n1] = pc2;
		      
		      /* And bond is reciprocated */
		      for (n2 = 0; n2 < pc2->s.nbonds2; n2++) {
			if (pc2->s.bond2[n2] == pc1->s.index) {
			  nbondpair += 1;
			  pc2->bonded2[n2] = pc1;
			}
		      }
		    }
		  }

		  /* Cell list */
	        }
	      }
	    }
	  }
	}
      }
    }
  }

  if (nbondfound != nbondpair) {
    /* There is a mismatch in the bond information (treat as fatal) */
    pe_fatal(obj->pe, "Find bonds2: bond not reciprocated\n");
  }

  return 0;
}



/*****************************************************************************
 *
 *  interact_find_bonds3
 *
 *  For backwards compatability, the case where only local bonds are
 *  included (nextra = 0).
 *
 *****************************************************************************/

int interact_find_bonds3(interact_t * obj, colloids_info_t * cinfo) {

  assert(obj);
  assert(cinfo);

  interact_find_bonds_all3(obj, cinfo, 1);

  return 0;
}

/*****************************************************************************
 *
 *  interact_find_bonds_all3
 *
 *  Examine the local colloids and match any bonded interactions
 *  in terms of pointers.
 *
 *  Include nextra cells in each direction into the halo region.
 *
 *****************************************************************************/

int interact_find_bonds_all3(interact_t * obj, colloids_info_t * cinfo,
			    int nextra) {

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int n1, n2;
  int ncell[3];
  
  int nbondfound = 0;
  int nbondpair = 0;

  colloid_t * pc1;
  colloid_t * pc2;

  assert(obj);
  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1 - nextra; ic1 <= ncell[X] + nextra; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1 - nextra; jc1 <= ncell[Y] + nextra; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1 - nextra; kc1 <= ncell[Z] + nextra; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {
	  if (pc1->s.nbonds3 == 0) continue;

	  for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
	    for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
	      for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
   
		colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &pc2);
		for (; pc2; pc2 = pc2->next) {

		  if (pc2->s.nbonds3 == 0) continue; 
                  
		  for (n1 = 0; n1 < pc1->s.nbonds3; n1++) {
		    if (pc1->s.bond3[n1] == pc2->s.index) {
		      nbondfound += 1;
		      pc1->bonded3[n1] = pc2;

		      /* And bond is reciprocated */
		      for (n2 = 0; n2 < pc2->s.nbonds3; n2++) {

			if (pc2->s.bond3[n2] == pc1->s.index) {
			  nbondpair += 1;
			  pc2->bonded3[n2] = pc1;

			}
		      }
		    }
		  }

		  /* Cell list */
	        }
	      }
	    }
	  }
	}
      }
    }
  }

  if (nbondfound != nbondpair) {
    /* There is a mismatch in the bond information (treat as fatal) */
    pe_fatal(obj->pe, "Find bonds3: bond not reciprocated\n");
  }

  return 0;
}





/*****************************************************************************
 *
 *  interact_rcmax
 *
 *  Return maximum centre-centre potential cut off
 *
 *****************************************************************************/

int interact_rcmax(interact_t * obj, double * rcmax) {

  int n;
  double rc = 0.0;

  assert(obj);

  for (n = 0; n < INTERACT_MAX; n++) {
    if (obj->rcset[n]) rc = dmax(rc, obj->rc[n]);
  }

  *rcmax = rc;

  return 0;
}

/*****************************************************************************
 *
 *  interaact_hcmax
 *
 *  Return maximum surface-surface cut off registered.
 *
 *****************************************************************************/

int interact_hcmax(interact_t * obj, double * hcmax) {

  int n;
  double hc = 0.0;

  assert(obj);
  assert(hcmax);

  for (n = 0; n < INTERACT_MAX; n++) {
    if (obj->hcset[n]) hc = dmax(hc, obj->hc[n]);
  }

  *hcmax = hc;

  return 0;
}

/*****************************************************************************
 *
 *  interact_range_check
 *
 *  Check the cell list width against the current interaction cut-off
 *  lengths.
 *
 *  For surface-surface separation based potentials, the criterion has
 *  a contribution of the largest colloid diameter present. For centre-
 *  centre calculations (such as Yukawa), this is not required.
 *
 *****************************************************************************/

int interact_range_check(interact_t * obj, colloids_info_t * cinfo) {

  int nc;
  int ncell[3];

  double ahmax;
  double rmax = 0.0;
  double lmin = DBL_MAX;
  double hc, rc;

  double lcell[3];

  assert(obj);
  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);
  if (nc < 2) return 0;

  colloids_info_lcell(cinfo, lcell);

  /* Work out the maximum cut-off range rmax */

  colloids_info_ahmax(cinfo, &ahmax);
  interact_rcmax(obj, &rc);
  interact_hcmax(obj, &hc);
  rmax = dmax(2.0*ahmax + hc, rc);

  /* Check against the cell list */

  colloids_info_ncell(cinfo, ncell);

  lmin = dmin(lmin, lcell[X]);
  lmin = dmin(lmin, lcell[Y]);
  lmin = dmin(lmin, lcell[Z]);

  if (rmax > lmin) {
    pe_info(obj->pe,
	    "Cell list width too small to capture specified interactions!\n");
    pe_info(obj->pe, "The maximum interaction range is: %f\n", rmax);
    pe_info(obj->pe, "The minumum cell width is only:   %f\n", lmin);
    pe_fatal(obj->pe, "Please check and try again\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_update_forces_ext
 *
 *  Having computed external forces, transfer the information for
 *  all particles to fex, tex.
 *
 *****************************************************************************/

int colloids_update_forces_ext(colloids_info_t * cinfo) {

  colloid_t * pc = NULL;

  assert(cinfo);

  colloids_info_all_head(cinfo, &pc);

  for (; pc; pc = pc->nextall) {
    pc->fex[X] = pc->force[X];
    pc->fex[Y] = pc->force[Y];
    pc->fex[Z] = pc->force[Z];

    pc->tex[X] = pc->torque[X];
    pc->tex[Y] = pc->torque[Y];
    pc->tex[Z] = pc->torque[Z];
 }

  return 0;
}
