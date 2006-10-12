/*****************************************************************************
 *
 *  active.c
 *
 *  Routines dealing with bounce-back on links for active particles.
 *
 *  $Id: active.c,v 1.1 2006-10-12 13:55:21 kevin Exp $
 *
 *  Isaac Llopis (Barcelona) developed the active particles.
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
#include "lattice.h"

enum active_type {TYPE_INACTIVE, TYPE_ONE_BIPOLAR, TYPE_ONE_QUADRUPOLAR}; 

static int    type_ = TYPE_INACTIVE;   /* Default */
static double is_quad_ = 0.0;          /* Unity if quadrupolar selected */

static void init_active1(void);
static void active1_prepass(void);

/*****************************************************************************
 *
 *  init_active
 *
 *****************************************************************************/

void init_active() {

  char tmp[128];

  RUN_get_string_parameter("colloid_type", tmp);

  /* Determine the request */

  if (strcmp(tmp, "active1_bipolar") == 0) {
    type_ = TYPE_ONE_BIPOLAR;
    info("\n\n[User   ] Active particle type one bipolar\n");
  }

  if (strcmp(tmp, "active1_quadrupolar") == 0) {
    type_ = TYPE_ONE_QUADRUPOLAR;
    info("\n\n[User   ] Active particle type one quadrulpolar\n");
    is_quad_ = 1.0;
  }

  switch (type_) {
  case TYPE_ONE_BIPOLAR:
  case TYPE_ONE_QUADRUPOLAR:
    init_active1();
    break;
  default:
    break;
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
  case TYPE_ONE_BIPOLAR:
  case TYPE_ONE_QUADRUPOLAR:
    active1_prepass();
    break;
  default:
    break;
  }

  return;
}

/*****************************************************************************
 *
 *  init_active1
 *
 *  Initise the cone angle and the momentum transfer parameter.
 *
 *****************************************************************************/

static void init_active1() {

  Colloid * p_colloid;
  int       n, ic, jc, kc;
  double    dp = 0.2, cone = 90;

  /* Read cone angle (degrees) and convert to radians */

  n = RUN_get_double_parameter("colloid_cone_angle", &cone);
  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("active cone angle (degrees) %f\n", cone);

  cone *= (PI/180.0);

  n = RUN_get_double_parameter("colloid_dp", &dp);
  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("active momentum parameter %f\n", dp);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {             
	  p_colloid->dp = dp;
	  p_colloid->cosine_ca = cos(cone);
	  /* Initialise direction vector */
	  p_colloid->dir.x = 0.0;
	  p_colloid->dir.y = 0.0;
	  p_colloid->dir.z = -1.0;

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
 *  active1_prepass
 *
 *  This step is called before the collision for "active1" particles.
 *
 *****************************************************************************/

static void active1_prepass() {

  Colloid   * p_colloid;
  COLL_Link * p_link;
  int         ic, jc, kc;
  double      deltap[3], force[3];
  double      rdots;

  /* Count the links lying within the cone for each colloid. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {             
 
	  /* Count the links */

	  p_link = p_colloid->lnk;
	  p_colloid->n1_nodes = 0;
	  p_colloid->n2_nodes = 0; 
   
	  while (p_link != NULL) {
     
	    if (p_link->status == LINK_FLUID) {
	      rdots = UTIL_dot_product(p_link->rb, p_colloid->dir)
		/ UTIL_fvector_mod(p_link->rb);

	      if (rdots >  p_colloid->cosine_ca) p_colloid->n1_nodes++; 
	      if (rdots < -p_colloid->cosine_ca) p_colloid->n2_nodes++;
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}
	/* Next cell */
      }
    }
  }

  CCOM_halo_sum(CHALO_TYPE6);

  /* Set the force on the fluid from each link */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {

	  deltap[X] = p_colloid->dp * p_colloid->dir.x;
	  deltap[Y] = p_colloid->dp * p_colloid->dir.y;
	  deltap[Z] = p_colloid->dp * p_colloid->dir.z;

	  p_link = p_colloid->lnk;

	  while (p_link != NULL) {
     
	    if (p_link->status == LINK_FLUID) {
    
	      rdots = UTIL_dot_product(p_link->rb, p_colloid->dir)
		/ UTIL_fvector_mod(p_link->rb);

	      if (rdots > p_colloid->cosine_ca) {
		force[X] = -deltap[X] / p_colloid->n1_nodes;
		force[Y] = -deltap[Y] / p_colloid->n1_nodes;
		force[Z] = -deltap[Z] / p_colloid->n1_nodes;    
		add_force_at_site(p_link->i, force);
	      }

	      if (rdots < -p_colloid->cosine_ca) {
		force[X] = -is_quad_*(deltap[X] / p_colloid->n2_nodes);
		force[Y] = -is_quad_*(deltap[Y] / p_colloid->n2_nodes);
		force[Z] = -is_quad_*(deltap[Z] / p_colloid->n2_nodes);
		add_force_at_site(p_link->i, force);
	      }
	    }

	    /* Next link */
	    p_link = p_link->next;
	  }
	  /* Next colloid */
	  p_colloid = p_colloid->next;	
	}
	/* Next cell */
      }
    }
  }
    
  /* Finally, set the force on each particle */

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid != NULL) {

	  p_colloid->f0.x += (1.0 + is_quad_)*p_colloid->dp*p_colloid->dir.x;
	  p_colloid->f0.y += (1.0 + is_quad_)*p_colloid->dp*p_colloid->dir.y;
	  p_colloid->f0.z += (1.0 + is_quad_)*p_colloid->dp*p_colloid->dir.z;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

