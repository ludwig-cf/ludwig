/*****************************************************************************
 *
 *  colloids_init.c
 *
 *  A very simple initialisation routine which can be used for
 *  small numbers of particles, which are placed at random.
 *  If there are any collisions in the result, a fatal error
 *  is issued.
 *
 *  Anything more complex should be organised separately and
 *  initialised from file.
 *
 *  $Id: colloids_init.c,v 1.3 2010-10-21 18:13:42 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>
#include <assert.h>

#include "util.h"
#include "coords.h"
#include "colloids_halo.h"
#include "colloids_init.h"
#include "wall.h"

static int colloids_init_check_state(colloids_info_t * cinfo, double hmax);
static int colloids_init_random_set(colloids_info_t * cinfo, int n,
				    const colloid_state_t * s,  double amax);
static int colloids_init_check_wall(colloids_info_t * cinfo, double dh);

/*****************************************************************************
 *
 *  colloids_init_random
 *
 *  Run the initialisation with a total of np particles.
 *
 *****************************************************************************/

int colloids_init_random(colloids_info_t * cinfo, int np,
			 const colloid_state_t * s0, double dh) {
  double amax;
  double hmax;

  assert(cinfo);

  /* Assume maximum size set by ah and small separation dh */
  amax = s0->ah + dh;
  hmax = 2.0*s0->ah + dh;

  colloids_init_random_set(cinfo, np, s0, amax);
  colloids_halo_state(cinfo);
  colloids_init_check_state(cinfo, hmax);

  if (wall_present()) colloids_init_check_wall(cinfo, dh);
  colloids_info_ntotal_set(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_random_set
 *
 *  Initialise a fixed number of particles in random positions.
 *  This is serial, and does not prevent collisions.
 *
 *****************************************************************************/

static int colloids_init_random_set(colloids_info_t * cinfo, int npart,
				    const colloid_state_t * s,  double amax) {
  int n;
  int state = 13;
  double r0[3];
  double lex[3];
  double ran[3];
  colloid_t * pc;

  /* If boundaries are not perioidic, some of the volume must be excluded */

  lex[X] = amax*(1.0 - is_periodic(X));
  lex[Y] = amax*(1.0 - is_periodic(Y));
  lex[Z] = amax*(1.0 - is_periodic(Z));

  for (n = 1; n <= npart; n++) {
    util_ranlcg_reap_uniform(&state, ran + X);
    util_ranlcg_reap_uniform(&state, ran + Y);
    util_ranlcg_reap_uniform(&state, ran + Z);

    r0[X] = Lmin(X) + lex[X] + ran[X]*(L(X) - 2.0*lex[X]);
    r0[Y] = Lmin(Y) + lex[Y] + ran[Y]*(L(Y) - 2.0*lex[Y]);
    r0[Z] = Lmin(Z) + lex[Z] + ran[Z]*(L(Z) - 2.0*lex[Z]);
    colloids_info_add_local(cinfo, n, r0, &pc);

    if (pc) {
      /* Copy the state in, except the index and position, and rebuild */
      pc->s = *s;
      pc->s.index = n;
      pc->s.rng = n;
      pc->s.rebuild = 1;
      pc->s.r[X] = r0[X];
      pc->s.r[Y] = r0[Y];
      pc->s.r[Z] = r0[Z];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_check_state
 *
 *  Check there are no hard sphere overlaps with centre-centre
 *  separation < dhmax.
 *
 *****************************************************************************/

static int colloids_init_check_state(colloids_info_t * cinfo, double hmax) {

  int noverlap_local;
  int noverlap;
  int ic, jc, kc, id, jd, kd, dx, dy, dz;
  int ncell[3];
  double hh;
  double r12[3];

  colloid_t * p_c1;
  colloid_t * p_c2;

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  noverlap_local = 0;

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &p_c1);

	while (p_c1) {
	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;
		colloids_info_cell_list_head(cinfo, id, jd, kd, &p_c2);

		while (p_c2) {
		  if (p_c2 != p_c1) {
		    coords_minimum_distance(p_c1->s.r, p_c2->s.r, r12);
		    hh = r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z];
		    if (hh < hmax*hmax) noverlap_local += 1;
		  }
		  /* Next colloid c2 */
		  p_c2 = p_c2->next;
		}
		/* Next search cell */
	      }
	    }
	  }
	  /* Next colloid c1 */
	  p_c1 = p_c1->next;
	}
	/* Next cell */
      }
    }
  }

  MPI_Allreduce(&noverlap_local, &noverlap, 1, MPI_INT, MPI_SUM, pe_comm());

  if (noverlap > 0) {
    info("This appears to include at least one hard sphere overlap.\n");
    info("Please check the colloid parameters and try again\n");
    fatal("Stop.\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_check_wall
 *
 *  If the boundary is not periodic, assume there is a wall at coordinate
 *  position at Lmin and Lmax.
 *
 *  An additional excluded volume of width dh is allowed.
 *
 *****************************************************************************/

static int colloids_init_check_wall(colloids_info_t * cinfo, double dh) {

  int ic, jc, kc, ia;
  int ncell[3];
  int ifailocal = 0;
  int ifail;

  colloid_t * pc = NULL;

  assert(cinfo);
  assert(dh >= 0.0);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  for (ia = 0; ia < 3; ia++) {
	    if (pc->s.r[ia] <= Lmin(ia) + pc->s.ah + dh) ifailocal = 1;
	    if (pc->s.r[ia] >= Lmin(ia) + L(ia) - pc->s.ah - dh) ifailocal = 1;
	  }
	  pc = pc->next;
	}

      }
    }
  }

  MPI_Allreduce(&ifailocal, &ifail, 1, MPI_INT, MPI_SUM, pe_comm());

  if (ifail) fatal("Colloid initial position overlaps wall\n");

  return 0;
}

