//CHANGE1
/*****************************************************************************
 *
 *  bond_harmonic3.c
 *
 *  Finite extensible elastic bond
 *
 *   V(r) = (1/2) k (r - r0)^2    
 *
 *  where r is the separation of bonded pairs.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Kai Qi (kai.qi@epfl.ch)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "bond_harmonic3.h"

struct bond_harmonic3_s {
  pe_t * pe;           /* Parallel environment */
  cs_t * cs;           /* Coordinate system */
  double k;            /* spring constant */
  double r0;           /* natrual length */
  double vlocal;       /* Bond potential contribution */
  double rminlocal;    /* Minimum bond extent */
  double rmaxlocal;    /* Maximum bond extension */
  double bondlocal;    /* Number of bonds computed (double) */
};

/*****************************************************************************
 *
 *  bond_harmonic3_create
 *
 *****************************************************************************/

int bond_harmonic3_create(pe_t * pe, cs_t * cs, bond_harmonic3_t ** pobj) {

  bond_harmonic3_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (bond_harmonic3_t *) calloc(1, sizeof(bond_harmonic3_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(bond_harmonic3_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_free
 *
 *****************************************************************************/

int bond_harmonic3_free(bond_harmonic3_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_param_set
 *
 *****************************************************************************/

int bond_harmonic3_param_set(bond_harmonic3_t * obj, double k, double r0) {

  assert(obj);

  obj->k = k;
  obj->r0 = r0;

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_info
 *
 *****************************************************************************/

int bond_harmonic3_info(bond_harmonic3_t * obj) {

  assert(obj);

  pe_info(obj->pe, "Harmonic bond\n");
  pe_info(obj->pe, "Spring constant:             %14.7e\n", obj->k);
  pe_info(obj->pe, "Equilibrium separation:      %14.7e\n", obj->r0);

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_register
 *
 *****************************************************************************/

int bond_harmonic3_register(bond_harmonic3_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_BOND_HARMONIC3, obj, bond_harmonic3_compute);
  interact_statistic_add(parent, INTERACT_BOND_HARMONIC3, obj, bond_harmonic3_stats);

/* -----> CHEMOVESICLE V2 */
/* Vesicle size constraint on cell width is now done by colloids_rt.c */
  interact_rc_set(parent, INTERACT_BOND_HARMONIC3, obj->r0);
/* <----- */

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_compute
 *
 *****************************************************************************/

int bond_harmonic3_compute(colloids_info_t * cinfo, void * self) {

  bond_harmonic3_t * obj = (bond_harmonic3_t *) self;

  int n;
  double r12[3];
  double r2min, r2max;
  double r2,f;
  double r2md;

  colloid_t * pc = NULL;

  assert(cinfo);
  assert(obj);

  colloids_info_local_head(cinfo, &pc);

  r2min = 4*obj->r0*obj->r0;
  r2max = 0.0;

  obj->vlocal = 0;
  obj->bondlocal = 0.0;

  for (; pc; pc = pc->nextlocal) {
   
    if (pc->s.nbonds3 == 0) continue;

    for (n = 0; n < pc->s.nbonds3; n++) {
      assert(pc->bonded3[n]);
      if (pc->s.index > pc->bonded3[n]->s.index) continue;

      /* Compute force arising on each particle from single bond */

      cs_minimum_distance(obj->cs, pc->s.r, pc->bonded3[n]->s.r, r12);
      r2 = r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z];
      r2md = sqrt(r2);

      if (r2 < r2min) r2min = r2;
      if (r2 > r2max) r2max = r2;
      if (r2 > 4*obj->r0*obj->r0) pe_fatal(obj->pe, "Broken harmonic3 bond\n");

      obj->vlocal += 0.5*obj->k*(r2md-obj->r0)*(r2md-obj->r0);
      obj->bondlocal += 1.0;
      f = -obj->k*(r2md-obj->r0)/r2md;

      pc->force[X] -= f*r12[X];
      pc->force[Y] -= f*r12[Y];
      pc->force[Z] -= f*r12[Z];

      pc->bonded3[n]->force[X] += f*r12[X];
      pc->bonded3[n]->force[Y] += f*r12[Y];
      pc->bonded3[n]->force[Z] += f*r12[Z];
    }
  }

  obj->rminlocal = sqrt(r2min);
  obj->rmaxlocal = sqrt(r2max);

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_stats
 *
 *****************************************************************************/

int bond_harmonic3_stats(void * self, double * stats) {

  bond_harmonic3_t * obj = (bond_harmonic3_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL]    = obj->vlocal;
  stats[INTERACT_STAT_RMINLOCAL] = obj->rminlocal;
  stats[INTERACT_STAT_RMAXLOCAL] = obj->rmaxlocal;

  return 0;
}

/*****************************************************************************
 *
 *  bond_harmonic3_single
 *
 *  For a single bond, compute v and |f| given r.
 *
 *****************************************************************************/

int bond_harmonic3_single(bond_harmonic3_t * obj, double r, double * v, double * f) {

  assert(obj);
  assert(r < 2*obj->r0);

  *v = 0.5*obj->k*(r-obj->r0)*(r-obj->r0);
  *f = -obj->k*(r-obj->r0);

  return 0;
}
