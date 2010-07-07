/*****************************************************************************
 *
 *  active.c
 *
 *  Routines dealing with bounce-back on links for active particles.
 *
 *  $Id: active.c,v 1.5.2.2 2010-07-07 09:00:31 kevin Exp $
 *
 *  Isaac Llopis (Barcelona) developed the active particles.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "runtime.h"

static void init_active2(void);

/*****************************************************************************
 *
 *  init_active
 *
 *****************************************************************************/

void init_active() {

  char tmp[128];

  RUN_get_string_parameter("colloid_type", tmp, 128);

  /* Determine the request */

  if (strcmp(tmp, "active2") == 0) {
    info("\n\n[User   ] Active particle type 2\n");
    init_active2();
  }

  return;
}

/*****************************************************************************
 *
 *  init_active2
 *
 *  Set the momentum transfer coefficient.
 *
 *****************************************************************************/

static void init_active2() {

  Colloid * p_colloid;
  int       n, ic, jc, kc;
  double    b_1 = 0.02, b_2 = 0.1;

  n = RUN_get_double_parameter("colloid_b1", &b_1);
  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("active B_1 parameter %f\n", b_1);
  n = RUN_get_double_parameter("colloid_b2", &b_2);
  info("active B_2 parameter %f\n", b_2);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid != NULL) {             
	  p_colloid->s.b1 = b_1;
	  p_colloid->s.b2 = b_2;
	  p_colloid->s.m[X] = 0.0;
	  p_colloid->s.m[Y] = 0.0;
	  p_colloid->s.m[Z] = -1.0;

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}
	/* Next cell */
      }
    }
  }

  return;
}
