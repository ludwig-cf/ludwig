/*****************************************************************************
 *
 *  active.c
 *
 *  Routines dealing with bounce-back on links for active particles.
 *
 *  $Id: active.c,v 1.5.2.1 2010-05-19 19:16:50 kevin Exp $
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
#include "ccomms.h"
#include "runtime.h"
#include "util.h"
#include "lattice.h"

enum active_type {TYPE_INACTIVE, TYPE_TWO}; 

static int    type_ = TYPE_INACTIVE;   /* Default */

static void init_active2(void);
static void active2_prepass(void);

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
    type_ = TYPE_TWO;
    info("\n\n[User   ] Active particle type 2\n");
    init_active2();
  }

  return;
}

/*****************************************************************************
 *
 *  active_bbl_prepass
 *
 *  Prepass to BBL called before the collision.
 *
 *****************************************************************************/

void active_bbl_prepass() {

  switch (type_) {
  case TYPE_INACTIVE:
    break;
  case TYPE_TWO:
    active2_prepass();
    break;
  default:
    break;
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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {             
	  p_colloid->b1 = b_1;
	  p_colloid->b2 = b_2;

	  /* Initialise direction vector */
	  p_colloid->direction[X] = 0.0;
	  p_colloid->direction[Y] = 0.0;
	  p_colloid->direction[Z] = -1.0;

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}
	/* Next cell */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  active2_prepass
 *
 *  This step is called for "active2" particles.
 *
 *****************************************************************************/

static void active2_prepass() {

  return;
}
