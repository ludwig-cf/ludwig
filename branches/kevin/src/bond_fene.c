/*****************************************************************************
 *
 *  bond_fene.c
 *
 *  Finite extensible non-linear elastic bond
 *
 *   V(r) = -(1/2) k r_0^2 ln [1 - (r/r_0)^2]    r < r_0
 *
 *  where r is the separation of bonded pairs.
 *
 *  Intended for use with LJ potential, where the parameters
 *  can be arranged to prevent bond-crossing, e.g.,
 *    r_0 = 1.5sigma  and k = 30epsilon / sigma^2
 *  cf. Kremer and Grest J. Chem. Phys. 92, 5057--5086 (1990).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "bond_fene.h"

struct bond_fene_s {
  int nref;            /* Reference counter */
  coords_t * cs;       /* Reference to coordinate system */
  double k;            /* 'spring' constant */
  double r0;           /* Maximum separation */
  double vlocal;       /* Bond potential contribution */
  double rminlocal;    /* Minimum bond extent */
  double rmaxlocal;    /* Maximum bond extension */
  double bondlocal;    /* Number of bonds computed (double) */
};

int bond_fene_release(void * self);

/*****************************************************************************
 *
 *  bond_fene_create
 *
 *****************************************************************************/

int bond_fene_create(coords_t * cs, bond_fene_t ** pobj) {

  bond_fene_t * obj = NULL;

  assert(pobj);

  obj = (bond_fene_t *) calloc(1, sizeof(bond_fene_t));
  if (obj == NULL) fatal("calloc(bond_fene_t) failed\n");

  obj->nref = 1;
  obj->cs = cs;
  coords_retain(cs);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_free
 *
 *****************************************************************************/

int bond_fene_free(bond_fene_t * obj) {

  if (obj) {
    obj->nref -= 1;
    if (obj->nref <= 0 ) {
      coords_free(obj->cs);
      free(obj);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_release
 *
 *****************************************************************************/

int bond_fene_release(void * self) {

  bond_fene_t * obj = (bond_fene_t *) self;

  if (obj) bond_fene_free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_param_set
 *
 *****************************************************************************/

int bond_fene_param_set(bond_fene_t * obj, double k, double r0) {

  assert(obj);

  obj->k = k;
  obj->r0 = r0;

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_info
 *
 *****************************************************************************/

int bond_fene_info(bond_fene_t * obj) {

  assert(obj);

  info("FENE bond\n");
  info("Spring constant:             %14.7e\n", obj->k);
  info("Equilibrium separation:      %14.7e\n", obj->r0);

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_register
 *
 *****************************************************************************/

int bond_fene_register(bond_fene_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_BOND, obj, bond_fene_compute);
  interact_statistic_add(parent, INTERACT_BOND, obj, bond_fene_stats);
  interact_rc_set(parent, INTERACT_BOND, obj->r0);
  interact_release_add(parent, INTERACT_BOND, obj, bond_fene_release);

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_compute
 *
 *****************************************************************************/

int bond_fene_compute(colloids_info_t * cinfo, void * self) {

  bond_fene_t * obj = (bond_fene_t *) self;

  int n;
  double r12[3];
  double r2min, r2max;
  double r2, rr02, f;

  colloid_t * pc = NULL;

  assert(cinfo);
  assert(obj);

  colloids_info_local_head(cinfo, &pc);
  rr02 = 1.0/(obj->r0*obj->r0);

  r2min = obj->r0*obj->r0;
  r2max = 0.0;

  obj->vlocal = 0;
  obj->bondlocal = 0.0;

  for (; pc; pc = pc->nextlocal) {

    for (n = 0; n < pc->s.nbonds; n++) {
      assert(pc->bonded[n]);
      if (pc->s.index > pc->bonded[n]->s.index) continue;

      /* Compute force arising on each particle from single bond */

      coords_minimum_distance(obj->cs, pc->s.r, pc->bonded[n]->s.r, r12);
      r2 = r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z];

      if (r2 < r2min) r2min = r2;
      if (r2 > r2max) r2max = r2;
      if (r2 > obj->r0*obj->r0) fatal("Broken fene bond\n");

      obj->vlocal += -0.5*obj->k*obj->r0*obj->r0*log(1.0 - r2*rr02);
      obj->bondlocal += 1.0;
      f = -obj->k/(1.0 - r2*rr02);

      pc->force[X] -= f*r12[X];
      pc->force[Y] -= f*r12[Y];
      pc->force[Z] -= f*r12[Z];

      pc->bonded[n]->force[X] += f*r12[X];
      pc->bonded[n]->force[Y] += f*r12[Y];
      pc->bonded[n]->force[Z] += f*r12[Z];
    }

  }

  obj->rminlocal = sqrt(r2min);
  obj->rmaxlocal = sqrt(r2max);

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_stats
 *
 *****************************************************************************/

int bond_fene_stats(void * self, double * stats) {

  bond_fene_t * obj = (bond_fene_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL]    = obj->vlocal;
  stats[INTERACT_STAT_RMINLOCAL] = obj->rminlocal;
  stats[INTERACT_STAT_RMAXLOCAL] = obj->rmaxlocal;

  return 0;
}

/*****************************************************************************
 *
 *  bond_fene_single
 *
 *  For a single bond, compute v and |f| given r.
 *
 *****************************************************************************/

int bond_fene_single(bond_fene_t * obj, double r, double * v, double * f) {

  double rr0;

  assert(obj);
  assert(r < obj->r0);

  rr0 = r/obj->r0;
  *v = -0.5*obj->k*obj->r0*obj->r0*log(1.0 - rr0*rr0);
  *f = -obj->k*r/(1.0 - rr0*rr0);

  return 0;
}
