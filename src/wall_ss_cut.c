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
 *  (c) 2010-2016 The University of Edinburgh
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
  double rminlocal;      /* local min centre-centre separation */
};

/*****************************************************************************
 *
 *  wall_ss_cut_create
 *
 *****************************************************************************/

int wall_ss_cut_create(pe_t * pe, cs_t * cs, wall_t * wall,
		       wall_ss_cut_t ** pobj) {

  wall_ss_cut_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (wall_ss_cut_t *) calloc(1, sizeof(wall_ss_cut_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(wall_ss_cut_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->wall = wall;

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
 *  wall_ss_cut_param_set
 *
 *****************************************************************************/

int wall_ss_cut_param_set(wall_ss_cut_t * obj, double epsilon, double sigma,
			  int nu, double hc) {
  assert(obj);

  obj->epsilon = epsilon;
  obj->sigma = sigma;
  obj->nu = nu;
  obj->hc = hc;

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
  pe_info(obj->pe, "Soft sphere for wall potential\n");
  pe_info(obj->pe, "epsilon:                  %14.7e\n", obj->epsilon);
  pe_info(obj->pe, "sigma:                    %14.7e\n", obj->sigma);
  pe_info(obj->pe, "exponent nu:              %14.7e\n", obj->nu);
  pe_info(obj->pe, "cut off (surface-surface) %14.7e\n", obj->hc);
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
  
  int ic1, jc1, kc1;
  int ncell[3];
  int ia;
  int iswall[3];
  
  double r;                             /* centre-centre sepration */
  double h;                             /* surface-surface separation */
  double rh;                            /* reciprocal h */
  double rsigma;                        /* reciproal sigma */
  double vcut;                          /* potential at cut off */
  double dvcut;                         /* derivative at cut off */
  double forcewall[3];                  /* force on the wall for accounting purposes */
  double lmin[3];
  double ltot[3];
  double f;

  colloid_t * pc1;
  
  assert(cinfo);
  assert(self);

  cs_lmin(self->cs, lmin);
  cs_ltot(self->cs, ltot);

  self->vlocal = 0.0;
  self->hminlocal = dmax(ltot[X], dmax(ltot[Y], ltot[Z]));
  self->rminlocal = self->hminlocal;

  rsigma = 1.0/self->sigma;
  vcut = self->epsilon*pow(self->sigma/self->hc, self->nu);
  dvcut = -self->epsilon*self->nu*rsigma*pow(self->sigma/self->hc, self->nu+1);

  forcewall[X] = 0.0;
  forcewall[Y] = 0.0;
  forcewall[Z] = 0.0;

  wall_present_dim(self->wall, iswall);
  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1; ic1 <= ncell[X]; ic1++) {
    for (jc1 = 1; jc1 <= ncell[Y]; jc1++) {
      for (kc1 = 1; kc1 <= ncell[Z]; kc1++) {
	
        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {

	  for (ia = 0; ia < 3; ia++) {
	    if (iswall[ia]) {
	      
	      f = 0.0;
	      /* lower wall */
	      r = pc1->s.r[ia] - lmin[ia];
	      if (r < self->rminlocal) self->rminlocal = r;
	      
	      h = r - pc1->s.ah;
	      assert(h > 0.0);
	      if (h < self->hminlocal) self->hminlocal = h;
	      
	      if (h < self->hc) {
		rh = 1.0/h;
		self->vlocal += self->epsilon*pow(rh*self->sigma, self->nu)
		  - vcut - (h - self->hc)*dvcut;
		f = -(-self->epsilon*self->nu*rsigma
		      *pow(rh*self->sigma, self->nu+1) - dvcut);
	      }
	      
	      /*upper wall*/
	      r = lmin[ia] + ltot[ia] - pc1->s.r[ia];
	      if (r < self->rminlocal) self->rminlocal = r;
	      
	      h = r - pc1->s.ah;
	      assert(h > 0.0);
	      if (h < self->hminlocal) self->hminlocal = h;
	      
	      if (h < self->hc) {
		rh = 1.0/h;
		self->vlocal += self->epsilon*pow(rh*self->sigma, self->nu)
		  - vcut - (h - self->hc)*dvcut;
		f -= -(-self->epsilon*self->nu*rsigma
		       *pow(rh*self->sigma, self->nu+1) - dvcut);
	      }
	      pc1->force[ia] += f;
	      forcewall[ia] -= f;
	    }
	  }
	}
      }
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
