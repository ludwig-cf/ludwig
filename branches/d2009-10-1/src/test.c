/*****************************************************************************
 *
 *  test.c
 *
 *  Some remaining statistics routines to be relocated.
 *
 *  $Id: test.c,v 1.19.4.1 2010-01-15 16:48:59 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "test.h"

/*****************************************************************************
 *
 *  test_colloid_momentum
 *
 *  Return net colloid momentum as g[3].
 *
 *  The final reduction to rank 0 in MPI_COMM_WORLD is for the
 *  purpose of output statistics.
 *
 *****************************************************************************/

void test_colloid_momentum(double g[3]) {

  int ic, jc, kc;

  double g_local[3];
  double mass;

  Colloid * p_colloid;

  g_local[X] = 0.0;
  g_local[Y] = 0.0;
  g_local[Z] = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  mass = 4.0*PI*pow(p_colloid->a0, 3)/3.0;

	  g_local[X] += mass*p_colloid->v.x;
	  g_local[Y] += mass*p_colloid->v.y;
	  g_local[Z] += mass*p_colloid->v.z;

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

  MPI_Reduce(g_local, g, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return;
}
