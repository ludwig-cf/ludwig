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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdlib.h>

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
		     map_t * map, psi_t * psi, ewald_t * ewald) {

  int nc;

  assert(interact);
  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);

  if (nc > 0) {
    colloids_update_forces_zero(cinfo);
    colloids_update_forces_external(cinfo, psi);
    colloids_update_forces_fluid_gravity(cinfo, map);
    colloids_update_forces_fluid_driven(cinfo, map);
    interact_wall(interact, cinfo);
    
    if (nc > 1) {
      interact_pairwise(interact, cinfo);
      interact_bonds(interact, cinfo);
      interact_angles(interact, cinfo);
      if (ewald) ewald_sum(ewald);
    }

    if (is_statistics_step()) {

      pe_info(interact->pe, "\nParticle statistics:\n");

      interact_stats(interact, cinfo);
      pe_info(interact->pe, "\n");
      stats_colloid_velocity_minmax(cinfo);
    }
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

	for (; pc != NULL; pc = pc->next) {
	  btorque[X] = pc->s.s[Y]*b0[Z] - pc->s.s[Z]*b0[Y];
	  btorque[Y] = pc->s.s[Z]*b0[X] - pc->s.s[X]*b0[Z];
	  btorque[Z] = pc->s.s[X]*b0[Y] - pc->s.s[Y]*b0[X];

	  driven_colloid_force(pc->s.s, dforce);
	  
	  for (ia = 0; ia < 3; ia++) {
	    pc->force[ia] += g[ia];                /* Gravity */
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
#ifdef OLD_SHIT
  int nc;
  int ia;
  int nsfluid;
  double rvolume;
  double fd[3], f[3];
  /* double fw[3]; */
  physics_t * phys = NULL;

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
      f[ia] = -1.0*fd[ia]*rvolume*is_periodic(ia);
      fw[ia] = -1.0*fd[ia]*(1.0 - is_periodic(ia))/(1.0*pe_size());
    }

    physics_fbody_set(phys, f);

    /* Need to account for wall momentum transfer */
    assert(0); /* NO TEST */
  }
#endif
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
  if (intr) interact_find_bonds(obj, cinfo);
  if (intr) obj->compute[INTERACT_BOND](cinfo, intr);

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

/*****************************************************************************
 *
 *  interact_find_bonds
 *
 *  Examine the local colloids and match any bonded interactions
 *  in terms of pointers.
 *
 *****************************************************************************/

int interact_find_bonds(interact_t * obj, colloids_info_t * cinfo) {

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

  for (ic1 = 1; ic1 <= ncell[X]; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1; jc1 <= ncell[Y]; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1; kc1 <= ncell[Z]; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {

          for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
            for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
              for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
   
                colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &pc2);
                for (; pc2; pc2 = pc2->next) {

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
