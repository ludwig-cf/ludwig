/*****************************************************************************
 *
 *  angle_cosine.c
 *
 *  Bending potential involving sites 1, 2, and 3
 *
 *     V(2) = kappa (r_1 - r_2).(r_3 - r_2) / | r_1 - r_2 | | r_3 - r_2 |
 *
 *          = kappa r_12 . r_32 / |r_12||r_32| = kappa cos(theta)
 *
 *   where theta is the angle at 2.
 *
 *   Edinburgh Soft Matter and Statistical Physics Group and
 *   Edinburgh Parallel Computing Centre
 *
 *   (c) The University of Edinburgh (2014)
 *   Contributing authors:
 *     Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "util.h"
#include "angle_cosine.h"

struct angle_cosine_s {
  int nref;
  coords_t * cs;
  double kappa;
  double vlocal;
  double cosine_max;
  double cosine_min;
};

int angle_cosine_release(void * self);

/*****************************************************************************
 *
 *  angle_cosine_create
 *
 *****************************************************************************/

int angle_cosine_create(coords_t * cs, angle_cosine_t ** pobj) {

  angle_cosine_t * obj = NULL;

  assert(pobj);

  obj = (angle_cosine_t *) calloc(1, sizeof(angle_cosine_t));
  if (obj == NULL) fatal("calloc(angle_cosine) failed\n");

  obj->nref = 1;
  obj->cs = cs;
  coords_retain(cs);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_free
 *
 *****************************************************************************/

int angle_cosine_free(angle_cosine_t * obj) {

  if (obj) {
    obj->nref -= 1;
    if (obj->nref <= 0) {
      coords_free(obj->cs);
      free(obj);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_release
 *
 *****************************************************************************/

int angle_cosine_release(void * self) {

  angle_cosine_t * angle = (angle_cosine_t *) self;

  if  (angle) angle_cosine_free(angle);

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_param_set
 *
 *****************************************************************************/

int angle_cosine_param_set(angle_cosine_t * obj, double kappa) {

  assert(obj);

  obj->kappa = kappa;

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_info
 *
 *****************************************************************************/

int angle_cosine_info(angle_cosine_t * obj) {

  assert(obj);

  info("Bond angle\n");
  info("Type:                         cosine\n");
  info("kappa:                       %14.7e\n", obj->kappa);

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_register
 *
 *****************************************************************************/

int angle_cosine_register(angle_cosine_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_ANGLE, obj, angle_cosine_compute);
  interact_statistic_add(parent, INTERACT_ANGLE, obj, angle_cosine_stats);
  interact_release_add(parent, INTERACT_ANGLE, obj, angle_cosine_release);

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_compute
 *
 *****************************************************************************/

int angle_cosine_compute(colloids_info_t * cinfo, void * self) {

  angle_cosine_t * obj = (angle_cosine_t *) self;

  double r0[3], r1[3];       /* separations */
  double r0sq, r1sq;         /* squared separations */
  double r0md, r1md;         /* moduli */
  double cosine;             /* of angle */
  double v0, v1, v01;        /* potential coefficients */
  double f0[3], f1[3];       /* forces */

  colloid_t * pc = NULL;

  assert(obj);
  assert(cinfo);

  obj->vlocal = 0.0;
  obj->cosine_min = +DBL_MAX;
  obj->cosine_max = -DBL_MAX;
  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {
    if (pc->s.nangles == 0) continue;
    assert(pc->s.nangles == 1);

    assert(pc->bonded[0]);
    assert(pc->bonded[1]);
    assert(pc->bonded[0]->s.index == pc->s.bond[0]);
    assert(pc->bonded[1]->s.index == pc->s.bond[1]);

    /* Bond 0 is pc -> bonded[0] */

    coords_minimum_distance(obj->cs, pc->s.r, pc->bonded[0]->s.r, r0);
    r0sq = r0[X]*r0[X] + r0[Y]*r0[Y] + r0[Z]*r0[Z];
    r0md = sqrt(r0sq);

    /* Bond 2 is pc -> bonded[1] */

    coords_minimum_distance(obj->cs, pc->s.r, pc->bonded[1]->s.r, r1);
    r1sq = r1[X]*r1[X] + r1[Y]*r1[Y] + r1[Z]*r1[Z];
    r1md = sqrt(r1sq);

    cosine = dot_product(r0, r1) / (r0md*r1md);
    assert(cosine <= 1.0);
    assert(cosine >= -1.0);

    v0  = obj->kappa*cosine/r0sq;
    v01 = obj->kappa/(r0md*r1md);
    v1  = obj->kappa*cosine/r1sq;

    f0[X] = v0*r0[X] - v01*r1[X];
    f0[Y] = v0*r0[Y] - v01*r1[Y];
    f0[Z] = v0*r0[Z] - v01*r1[Z];

    f1[X] = v1*r1[X] - v01*r0[X];
    f1[Y] = v1*r1[Y] - v01*r0[Y];
    f1[Z] = v1*r1[Z] - v01*r0[Z];

    /* Accumulate forces */

    pc->bonded[0]->force[X] += f0[X];
    pc->bonded[0]->force[Y] += f0[Y];
    pc->bonded[0]->force[Z] += f0[Z];

    pc->force[X] -= (f0[X] + f1[X]);
    pc->force[Y] -= (f0[Y] + f1[Y]);
    pc->force[Z] -= (f0[Z] + f1[Z]);

    pc->bonded[1]->force[X] += f1[X];
    pc->bonded[1]->force[Y] += f1[Y];
    pc->bonded[1]->force[Z] += f1[Z];

    /* Potential energy */

    obj->vlocal += obj->kappa*cosine;
    if (cosine < obj->cosine_min) obj->cosine_min = cosine;
    if (cosine > obj->cosine_max) obj->cosine_max = cosine;
  }

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_stats
 *
 *****************************************************************************/

int angle_cosine_stats(void * self, double * stats) {

  angle_cosine_t * obj = (angle_cosine_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL] = obj->vlocal;
  /* "rmax" "rmin" here are radians */
  stats[INTERACT_STAT_RMINLOCAL] = acos(obj->cosine_min);
  stats[INTERACT_STAT_RMAXLOCAL] = acos(obj->cosine_max);

  return 0;
}

