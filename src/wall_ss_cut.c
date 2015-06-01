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
 *  (c) 2010-2014 The University of Edinburgh
 *  Juho Lintuvuori (juho.lintuvuori@u-psud.fr)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "wall.h"
#include "wall_ss_cut.h"

struct wall_ss_cut_s {
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

int wall_ss_cut_create(wall_ss_cut_t ** pobj) {

  wall_ss_cut_t * obj = NULL;

  assert(pobj);

  obj = (wall_ss_cut_t *) calloc(1, sizeof(wall_ss_cut_t));
  if (obj == NULL) fatal("calloc(wall_ss_cut_t) failed\n");

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  wall_ss_cut_free
 *
 *****************************************************************************/

void wall_ss_cut_free(wall_ss_cut_t * obj) {

  assert(obj);

  free(obj);

  return;
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
  physics_kt(&kt);

  info("\n");
  info("Soft sphere for wall potential\n");
  info("epsilon:                  %14.7e\n", obj->epsilon);
  info("sigma:                    %14.7e\n", obj->sigma);
  info("exponent nu:              %14.7e\n", obj->nu);
  info("cut off (surface-surface) %14.7e\n", obj->hc);
  if (kt > 0.0) {
    info("epsilon / kT              %14.7e\n", obj->epsilon/kt);
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

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];

  int ia;
  
  double r;                             /* centre-centre sepration */
  double h;                             /* surface-surface separation */
  double rh;                            /* reciprocal h */
  double rsigma;                        /* reciproal sigma */
  double vcut;                          /* potential at cut off */
  double dvcut;                         /* derivative at cut off */
  double forcewall[3];                  /* force on the wall for accounting purposes */
  double f;

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(self);

  self->vlocal = 0.0;
  self->hminlocal = dmax(L(X), dmax(L(Y), L(Z)));
  self->rminlocal = self->hminlocal;

  rsigma = 1.0/self->sigma;
  vcut = self->epsilon*pow(self->sigma/self->hc, self->nu);
  dvcut = -self->epsilon*self->nu*rsigma*pow(self->sigma/self->hc, self->nu+1);

  forcewall[X] = 0.0;
  forcewall[Y] = 0.0;
  forcewall[Z] = 0.0;
  
  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1; ic1 <= ncell[X]; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1; jc1 <= ncell[Y]; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1; kc1 <= ncell[Z]; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {

	  for (ia = 0; ia < 3; ia++){
	    if(wall_at_edge(ia)){
	      
	      f = 0.0;
	      /* lower wall */
	      r = pc1->s.r[ia] - Lmin(ia);
	      if (r < self->rminlocal) self->rminlocal = r;
	      
	      h = r - pc1->s.ah;
	      assert(h > 0.0);
	      if (h < self->hminlocal) self->hminlocal = h;
	      
	      if (h < self->hc){
		rh = 1.0/h;
		self->vlocal += self->epsilon*pow(rh*self->sigma, self->nu)
		  - vcut - (h - self->hc)*dvcut;
		f = -(-self->epsilon*self->nu*rsigma
		      *pow(rh*self->sigma, self->nu+1) - dvcut);
	      }
	      
	      /*upper wall*/
	      r = Lmin(ia) + L(ia) - pc1->s.r[ia];
	      if (r < self->rminlocal) self->rminlocal = r;
	      
	      h = r - pc1->s.ah;
	      assert(h > 0.0);
	      if (h < self->hminlocal) self->hminlocal = h;
	      
	      if (h < self->hc){
		rh = 1.0/h;
		self->vlocal += self->epsilon*pow(rh*self->sigma, self->nu)
		  - vcut - (h - self->hc)*dvcut;
		f -= -(-self->epsilon*self->nu*rsigma
		       *pow(rh*self->sigma, self->nu+1) - dvcut); /*the sign armageddon matters*/
	      }
	      pc1->force[ia] += f;
	      forcewall[ia] -= f;
	    }
	  }
	}
      }
    }
  }

  wall_accumulate_force(forcewall);
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
