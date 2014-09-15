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
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>
#include <assert.h>

#include "pe.h"
#include "ran.h"
#include "coords.h"
#include "colloids.h"
#include "colloids_halo.h"
#include "colloids_init.h"
#include "wall.h"

static void colloids_init_check_state(double hmax);
static void colloids_init_random_set(int n, const colloid_state_t * s,
				     double amax);
static int  colloids_init_check_wall(double dh);

/*****************************************************************************
 *
 *  colloids_init_random
 *
 *  Run the initialisation with a total of np particles.
 *  If np = 1, use the position of the given particle;
 *  otherwise, use random positions.
 *
 *****************************************************************************/

void colloids_init_random(int np, const colloid_state_t * s0, double dh) {

  double amax;
  double hmax;
  colloid_t * pc;

  if (np == 1) {
    pc = colloid_add_local(1, s0->r);
    if (pc) {
      pc->s = *s0;
      pc->s.index = 1;
      pc->s.rebuild = 1;
    }
  }
  else {
    /* Assume maximum size set by ah and small separation dh */
    amax = s0->ah + dh;
    hmax = 2.0*s0->ah + dh;

    colloids_init_random_set(np, s0, amax);
    colloids_halo_state();
    colloids_init_check_state(hmax);
  }

  if (wall_present()) colloids_init_check_wall(dh);
  colloids_ntotal_set();

  return;
}

/*****************************************************************************
 *
 *  colloids_init_random_set
 *
 *  Initialise a fixed number of particles in random positions.
 *  This is serial, and does not prevent collisions.
 *
 *****************************************************************************/

static void colloids_init_random_set(int npart, const colloid_state_t * s,
				     double amax) {
  int n;
  double r0[3];
  double lex[3];
  colloid_t * pc;

  /* If boundaries are not perioidic, some of the volume must be excluded */

  lex[X] = amax*(1.0 - is_periodic(X));
  lex[Y] = amax*(1.0 - is_periodic(Y));
  lex[Z] = amax*(1.0 - is_periodic(Z));

  for (n = 1; n <= npart; n++) {
    r0[X] = Lmin(X) + lex[X] + ran_serial_uniform()*(L(X) - 2.0*lex[X]);
    r0[Y] = Lmin(Y) + lex[Y] + ran_serial_uniform()*(L(Y) - 2.0*lex[Y]);
    r0[Z] = Lmin(Z) + lex[Z] + ran_serial_uniform()*(L(Z) - 2.0*lex[Z]);
    pc = colloid_add_local(n, r0);

    if (pc) {
      /* Copy the state in, except the index and position, and rebuild */
      pc->s = *s;
      pc->s.index = n;
      pc->s.rebuild = 1;
      pc->s.r[X] = r0[X];
      pc->s.r[Y] = r0[Y];
      pc->s.r[Z] = r0[Z];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_init_check_state
 *
 *  Check there are no hard sphere overlaps with centre-centre
 *  separation < dhmax.
 *
 *****************************************************************************/

static void colloids_init_check_state(double hmax) {

  int noverlap_local;
  int noverlap;
  int ic, jc, kc, id, jd, kd, dx, dy, dz;
  double hh;
  double r12[3];

  colloid_t * p_c1;
  colloid_t * p_c2;

  noverlap_local = 0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_c1 = colloids_cell_list(ic, jc, kc);

	while (p_c1) {
	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;
		p_c2 = colloids_cell_list(id, jd, kd);

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

  return;
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

static int colloids_init_check_wall(double dh) {

  int ic, jc, kc, ia;
  int ifailocal = 0;
  int ifail;

  colloid_t * pc = NULL;

  assert(dh >= 0.0);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

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
