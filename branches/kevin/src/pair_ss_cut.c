/*****************************************************************************
 *
 *  pair_ss_cut.c
 *
 *  Cut-and-shifted soft sphere pair potential.
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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "physics.h"
#include "pair_ss_cut.h"

struct pair_ss_cut_s {
  int nref;              /* reference counter */
  coords_t * cs;         /* reference to the coordinate system */
  double epsilon;        /* epsilon (energy) */
  double sigma;          /* sigma (length) */
  double nu;             /* exponent */
  double hc;             /* cut-off */
  double vlocal;         /* local contribution to energy */
  double hminlocal;      /* local nearest separation */
  double rminlocal;      /* local min centre-centre separation */
};

int pair_ss_cut_release(void * self);

/*****************************************************************************
 *
 *  pair_ss_cut_create
 *
 *****************************************************************************/

int pair_ss_cut_create(coords_t * cs, pair_ss_cut_t ** pobj) {

  pair_ss_cut_t * obj = NULL;

  assert(cs);
  assert(pobj);

  obj = (pair_ss_cut_t *) calloc(1, sizeof(pair_ss_cut_t));
  if (obj == NULL) fatal("calloc(pair_ss_cut_t) failed\n");

  obj->nref = 1;
  obj->cs = cs;
  coords_retain(cs);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_free
 *
 *****************************************************************************/

int pair_ss_cut_free(pair_ss_cut_t * obj) {

  if (obj) {
    obj->nref -= 1;
    if (obj->nref <= 0) {
      coords_free(&obj->cs);
      free(obj);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_release
 *
 *  Decrement reference count; intended to be called from parent
 *  interaction object to allow clean-up.
 *
 *****************************************************************************/

int pair_ss_cut_release(void * self) {

  pair_ss_cut_t * obj = (pair_ss_cut_t *) self;

  if (obj) pair_ss_cut_free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_param_set
 *
 *****************************************************************************/

int pair_ss_cut_param_set(pair_ss_cut_t * obj, double epsilon, double sigma,
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
 *  pair_ss_cut_info
 *
 *****************************************************************************/

int pair_ss_cut_info(pair_ss_cut_t * obj) {

  physics_t * phys;
  double kt;

  physics_ref(&phys);
  physics_kt(&kt);

  info("\n");
  info("Soft sphere potential\n");
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
 *  pair_ss_cut_register
 *
 *****************************************************************************/

int pair_ss_cut_register(pair_ss_cut_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_PAIR, obj, pair_ss_cut_compute);
  interact_statistic_add(parent, INTERACT_PAIR, obj, pair_ss_cut_stats);
  interact_hc_set(parent, INTERACT_PAIR, obj->hc);
  interact_release_add(parent, INTERACT_PAIR, obj, pair_ss_cut_release);

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_compute
 *
 *****************************************************************************/

int pair_ss_cut_compute(colloids_info_t * cinfo, void * obj) {

  pair_ss_cut_t * self = (pair_ss_cut_t *) obj;

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];

  double r;                             /* centre-centre sepration */
  double h;                             /* surface-surface separation */
  double rh;                            /* reciprocal h */
  double rsigma;                        /* reciproal sigma */
  double vcut;                          /* potential at cut off */
  double dvcut;                         /* derivative at cut off */
  double r12[3];                        /* centre-centre min distance 1->2 */
  double f;
  double ltot[3];                       /* system size */

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(self);

  coords_ltot(self->cs, ltot);

  self->vlocal = 0.0;
  self->hminlocal = dmax(ltot[X], dmax(ltot[Y], ltot[Z]));
  self->rminlocal = self->hminlocal;

  rsigma = 1.0/self->sigma;
  vcut = self->epsilon*pow(self->sigma/self->hc, self->nu);
  dvcut = -self->epsilon*self->nu*rsigma*pow(self->sigma/self->hc, self->nu+1);

  colloids_info_ncell(cinfo, ncell);

  for (ic1 = 1; ic1 <= ncell[X]; ic1++) {
    colloids_info_climits(cinfo, X, ic1, di); 
    for (jc1 = 1; jc1 <= ncell[Y]; jc1++) {
      colloids_info_climits(cinfo, Y, jc1, dj);
      for (kc1 = 1; kc1 <= ncell[Z]; kc1++) {
        colloids_info_climits(cinfo, Z, kc1, dk);

        colloids_info_cell_list_head(cinfo, ic1, jc1, kc1, &pc1);
        for (; pc1; pc1 = pc1->next) {

          for (ic2 = di[0]; ic2 <= di[1]; ic2++) {
            for (jc2 = dj[0]; jc2 <= dj[1]; jc2++) {
              for (kc2 = dk[0]; kc2 <= dk[1]; kc2++) {
   
		colloids_info_cell_list_head(cinfo, ic2, jc2, kc2, &pc2);
                for (; pc2; pc2 = pc2->next) {

                  if (pc1->s.index >= pc2->s.index) continue;

		  coords_minimum_distance(self->cs, pc1->s.r, pc2->s.r, r12);
		  r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);
		  if (r < self->rminlocal) self->rminlocal = r;

		  h = r - pc1->s.ah - pc2->s.ah;
		  if (h < self->hminlocal) self->hminlocal = h;

		  if (h > self->hc) continue;
		  assert(h > 0.0);

		  rh = 1.0/h;

		  self->vlocal += self->epsilon*pow(rh*self->sigma, self->nu)
		    - vcut - (h - self->hc)*dvcut;
		  f = -(-self->epsilon*self->nu*rsigma
			*pow(rh*self->sigma, self->nu+1) - dvcut);

		  rh = 1.0/r;
		  pc1->force[X] -= f*r12[X]*rh;
		  pc1->force[Y] -= f*r12[Y]*rh;
		  pc1->force[Z] -= f*r12[Z]*rh;
		  pc2->force[X] += f*r12[X]*rh;
		  pc2->force[Y] += f*r12[Y]*rh;
		  pc2->force[Z] += f*r12[Z]*rh;
		}
	      }
	    }
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_stats
 *
 *****************************************************************************/

int pair_ss_cut_stats(void * obj, double * stat) {

  pair_ss_cut_t * self = (pair_ss_cut_t *) obj;

  assert(self);

  stat[INTERACT_STAT_VLOCAL]    = self->vlocal;
  stat[INTERACT_STAT_HMINLOCAL] = self->hminlocal;
  stat[INTERACT_STAT_RMINLOCAL] = self->rminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_single
 *
 *  Returns scalar part of force for given h, and potential.
 *
 *****************************************************************************/

int pair_ss_cut_single(pair_ss_cut_t * obj, double h, double * f, double * v) {

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
