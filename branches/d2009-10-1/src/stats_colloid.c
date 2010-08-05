/*****************************************************************************
 *
 *  stats_colloid.c
 *
 *  Some useful quantities concerning colloids.
 *
 *  $Id: stats_colloid.c,v 1.1.2.4 2010-08-05 15:29:37 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "util.h"
#include "stats_colloid.h"

/*****************************************************************************
 *
 *  stats_colloid_momentum
 *
 *  Return net colloid momentum as g[3].
 *
 *  The final reduction to rank 0 in pe_comm is for the
 *  purpose of output statistics via info().
 *
 *****************************************************************************/

void stats_colloid_momentum(double g[3]) {

  int ic, jc, kc;

  double glocal[3];
  double mass;

  Colloid * p_colloid;

  glocal[X] = 0.0;
  glocal[Y] = 0.0;
  glocal[Z] = 0.0;

  if (colloid_ntotal() == 0) {
    /* do nothing */
  }
  else {

    for (ic = 1; ic <= Ncell(X); ic++) {
      for (jc = 1; jc <= Ncell(Y); jc++) {
	for (kc = 1; kc <= Ncell(Z); kc++) {

	  p_colloid = colloids_cell_list(ic, jc, kc);

	  while (p_colloid) {
	    mass = 4.0*pi_*pow(p_colloid->s.a0, 3)/3.0;

	    glocal[X] += mass*p_colloid->s.v[X];
	    glocal[Y] += mass*p_colloid->s.v[Y];
	    glocal[Z] += mass*p_colloid->s.v[Z];

	    /* Next colloid */
	    p_colloid = p_colloid->next;
	  }

	  /* Next cell */
	}
      }
    }

    MPI_Reduce(glocal, g, 3, MPI_DOUBLE, MPI_SUM, 0, pe_comm());
  }

  return;
}
