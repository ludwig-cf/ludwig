/*****************************************************************************
 *
 *  wall_ss_cut.c
 *
 *  Cut-and-shifted soft sphere pair potential for wall.
 *
 *  In addition, separation is computed as surface-surface separation ('h')
 *  and the potential (and force) are smoothly matched at cut off 'hc'.
 *
 *  The basic potential is
 *
 *     vss(r) = epsilon (sigma/r)^nu
 *
 *  while the cut-and-shift version is
 *
 *     vss(r) - vss(rc) - (r - rc) . dvss/dr|_rc
 *
 *  The force is then
 *
 *     f_ij(r) = -(dvss/dr - dvss/dr|_rc) \hat{r}_ij
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *  Juho Lintuvuori (juho.lintuvuori@u-psud.fr)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "util.h"
#include "physics.h"
#include "wall_ss_cut.h"

struct wall_ss_cut_s {
  pe_t * pe;             /* parallel environment */
  cs_t * cs;             /* coordinate system */
  wall_t * wall;         /* wall information */

  double epsilon;        /* epsilon (energy) */
  double sigma;          /* sigma (length) */
  double nu;             /* exponent */
  double hc;             /* cut-off */
  double vlocal;         /* local contribution to energy */
  double hminlocal;      /* local nearest separation */
  double rminlocal;      /* local min wall-centre separation */
};

/*****************************************************************************
 *
 *  wall_ss_cut_create
 *
 *****************************************************************************/

int wall_ss_cut_create(pe_t * pe, cs_t * cs, wall_t * wall,
		       const wall_ss_cut_options_t * opts,
		       wall_ss_cut_t ** pobj) {

  wall_ss_cut_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(wall);
  assert(opts);
  assert(pobj);

  obj = (wall_ss_cut_t *) calloc(1, sizeof(wall_ss_cut_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(wall_ss_cut_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->wall = wall;

  obj->epsilon = opts->epsilon;
  obj->sigma = opts->sigma;
  obj->nu = opts->nu;
  obj->hc = opts->hc;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_free
 *
 *****************************************************************************/

int wall_ss_cut_free(wall_ss_cut_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_info
 *
 *****************************************************************************/

int wall_ss_cut_info(wall_ss_cut_t * obj) {

  physics_t * phys;
  double kt;

  physics_ref(&phys);
  physics_kt(phys, &kt);

  pe_info(obj->pe, "\n");
  pe_info(obj->pe, "Wall-colloid soft-sphere potential\n");
  pe_info(obj->pe, "----------------------------------\n");
  pe_info(obj->pe, "epsilon:                  %14.7e\n", obj->epsilon);
  pe_info(obj->pe, "sigma:                    %14.7e\n", obj->sigma);
  pe_info(obj->pe, "exponent nu:              %14.7e\n", obj->nu);
  pe_info(obj->pe, "cut off hc (wall-surface) %14.7e\n", obj->hc);
  if (kt > 0.0) {
    pe_info(obj->pe, "epsilon / kT              %14.7e\n", obj->epsilon/kt);
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_register
 *
 *****************************************************************************/

int wall_ss_cut_register(wall_ss_cut_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_WALL, obj, wall_ss_cut_compute);
  interact_statistic_add(parent, INTERACT_WALL, obj, wall_ss_cut_stats);
  interact_hc_set(parent, INTERACT_WALL, obj->hc);

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_compute
 *
 *****************************************************************************/

int wall_ss_cut_compute(colloids_info_t * cinfo, void * obj) {

  wall_ss_cut_t * self = (wall_ss_cut_t *) obj;
  
  double forcewall[3] = {0};        /* force on the wall for accounting */
  double lmin[3];
  double ltot[3];

  colloid_t * pc = NULL;
  
  assert(cinfo);
  assert(self);

  cs_lmin(self->cs, lmin);
  cs_ltot(self->cs, ltot);

  self->vlocal = 0.0;
  self->hminlocal = dmax(ltot[X], dmax(ltot[Y], ltot[Z]));
  self->rminlocal = self->hminlocal;

  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {

    for (int ia = 0; ia < 3; ia++) {

      double f = 0.0;
      double r = 0.0;                       /* wall-centre sepration */
      double h = 0.0;                       /* wall-surface separation */

      if (self->wall->param->isboundary[ia] == 0) continue;
	      
      /* lower wall */
      r = pc->s.r[ia] - lmin[ia];
      h = r - pc->s.ah;

      if (h < self->hminlocal) self->hminlocal = h;
      if (r < self->rminlocal) self->rminlocal = r;
	      
      if (h < self->hc) {
	double v = 0.0;
	double fl = 0.0;
	wall_ss_cut_single(self, h, &fl, &v);
	self->vlocal += v;
	f += fl;
      }
	      
      /* upper wall */
      r = lmin[ia] + ltot[ia] - pc->s.r[ia];
      h = r - pc->s.ah;

      if (r < self->rminlocal) self->rminlocal = r;
      if (h < self->hminlocal) self->hminlocal = h;
	      
      if (h < self->hc) {
	double v = 0.0;
	double fu = 0.0;
	wall_ss_cut_single(self, h, &fu, &v);
	self->vlocal += v;
	f -= fu;            /* upper wall gives -ve (repulsive) force */
      }
      pc->force[ia] += f;
      forcewall[ia] -= f;
    }
  }

  wall_momentum_add(self->wall, forcewall);

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_stats
 *
 *****************************************************************************/

int wall_ss_cut_stats(void * obj, double * stat) {

  wall_ss_cut_t * self = (wall_ss_cut_t *) obj;

  assert(self);

  stat[INTERACT_STAT_VLOCAL]    = self->vlocal;
  stat[INTERACT_STAT_HMINLOCAL] = self->hminlocal;
  stat[INTERACT_STAT_RMINLOCAL] = self->rminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_single
 *
 *  Returns scalar part of force for given h, and potential.
 *
 *****************************************************************************/

int wall_ss_cut_single(wall_ss_cut_t * obj, double h, double * f, double * v) {

  double rh;
  double rsigma;
  double vcut, dvcut;

  assert(obj);
  assert(h > 0.0);

  rh = 1.0/h;
  rsigma = 1.0/obj->sigma;
  vcut = obj->epsilon*pow(obj->sigma/obj->hc, obj->nu);
  dvcut = -obj->epsilon*obj->nu*rsigma*pow(obj->sigma/obj->hc, obj->nu+1);

  *v = obj->epsilon*pow(rh*obj->sigma, obj->nu) - vcut - (h - obj->hc)*dvcut;
  *f = -(-obj->epsilon*obj->nu*rsigma*pow(rh*obj->sigma, obj->nu+1) - dvcut);

  /* Remember 'f' is -(dv/dr - dv/dr|_rc) */

  return 0;
}
