/*****************************************************************************
 *
 *  pair_lj_cut.c
 *
 *  Lennard-Jones (cut-and-shift version). The basic potential is
 *
 *    v(r) = 4epsilon [ (sigma/r)^12 - (sigma/r)^6 ]
 *
 *  while the cut-and-shift version is
 *
 *    v(r) - v(rc) - (r - rc) dv/dr|_rc
 *
 *  ensuring both potential and force smoothly go to zero at cutoff rc.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contributing authors:
 *    Juho Lintuvuori (jlintuvu@ph.ed.ac.uk)
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "pair_lj_cut.h"

struct pair_lj_cut_s {
  double epsilon;
  double sigma;
  double rc;
  double vlocal;
  double hminlocal;
  double rminlocal;
};

/*****************************************************************************
 *
 *  pair_lj_cut_create
 *
 *****************************************************************************/

int pair_lj_cut_create(pair_lj_cut_t ** pobj) {

  pair_lj_cut_t * obj = NULL;

  assert(pobj);

  obj = (pair_lj_cut_t *) calloc(1, sizeof(pair_lj_cut_t));
  if (obj == NULL) fatal("calloc(pair_lj_cut_t) failed\n");

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ly_cut_free
 *
 *****************************************************************************/

void pair_lj_cut_free(pair_lj_cut_t * obj) {

  assert(obj);

  free(obj);

  return;
}

/*****************************************************************************
 *
 *  pair_lj_cut_param_set
 *
 *****************************************************************************/

int pair_lj_cut_param_set(pair_lj_cut_t * obj, double epsilon, double sigma,
			  double rc) {
  assert(obj);

  obj->epsilon = epsilon;
  obj->sigma = sigma;
  obj->rc = rc;

  return 0;
}

/*****************************************************************************
 *
 *  pair_lj_cut_info
 *
 *****************************************************************************/

int pair_lj_cut_info(pair_lj_cut_t * obj) {

  assert(obj);

  info("Lennard-Jones potential\n");
  info("epsilon:                  %14.7e\n", obj->epsilon);
  info("sigma:                    %14.7e\n", obj->sigma);
  info("cut off (centre-centre)   %14.7e\n", obj->rc);

  return 0;
}

/*****************************************************************************
 *
 *  pair_lj_cut_register
 *
 *****************************************************************************/

int pair_lj_cut_register(pair_lj_cut_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_PAIR, obj, pair_lj_cut_compute);
  interact_statistic_add(parent, INTERACT_PAIR, obj, pair_lj_cut_stats);
  interact_rc_set(parent, INTERACT_PAIR, obj->rc);

  return 0;
}

/*****************************************************************************
 *
 *  pair_lj_cut_compute
 *
 *****************************************************************************/

int pair_lj_cut_compute(colloids_info_t * cinfo, void * self) {

  pair_lj_cut_t * obj = (pair_lj_cut_t *) self;

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];

  double r2;
  double r;
  double rr;
  double rs;
  double vcut;
  double dvcut;
  double r12[3];
  double f, h;

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(self);

  colloids_info_ncell(cinfo, ncell);

  obj->vlocal = 0.0;
  obj->rminlocal = dmax(L(X), dmax(L(Y), L(Z)));
  obj->hminlocal = obj->rminlocal;

  rr = 1.0/obj->rc;
  rs = pow(obj->sigma*rr, 6);

  vcut = 4.0*obj->epsilon*(rs*rs - rs);
  dvcut = -24.0*rr*obj->epsilon*(2.0*rs*rs - rs);

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

		  coords_minimum_distance(pc1->s.r, pc2->s.r, r12);
		  r2 = r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z];

		  r = sqrt(r2);

		  /* Record both rmin and hmin */
		  if (r < obj->rminlocal) obj->rminlocal = r;
		  h = r - pc1->s.ah -pc2->s.ah;
		  if (h < obj->hminlocal) obj->hminlocal = h;

		  if (r > obj->rc) continue;

		  rr = 1.0/r;
		  rs = pow(obj->sigma*rr, 6);

		  /* Potential, force */

		  obj->vlocal += 4.0*obj->epsilon*(rs*rs - rs) - vcut
		    - (r - obj->rc)*dvcut;
		  f = -(-24.0*rr*obj->epsilon*(2.0*rs*rs - rs) - dvcut);

		  pc1->force[X] -= f*r12[X]*rr;
		  pc1->force[Y] -= f*r12[Y]*rr;
		  pc1->force[Z] -= f*r12[Z]*rr;
		  pc2->force[X] += f*r12[X]*rr;
		  pc2->force[Y] += f*r12[Y]*rr;
		  pc2->force[Z] += f*r12[Z]*rr;

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
 *  pair_lj_cut_stats
 *
 *****************************************************************************/

int pair_lj_cut_stats(void * obj, double * stats) {

  pair_lj_cut_t * self = (pair_lj_cut_t *) obj;

  assert(self);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL]    = self->vlocal;
  stats[INTERACT_STAT_HMINLOCAL] = self->hminlocal;
  stats[INTERACT_STAT_RMINLOCAL] = self->rminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  pair_lj_cut_single
 *
 *****************************************************************************/

int pair_lj_cut_single(pair_lj_cut_t * obj, double r, double * f, double * v) {

  double rr;
  double rs;
  double vcut;
  double dvcut;

  assert(obj);
  assert(r > 0.0);

  rr = 1.0/obj->rc;
  rs = pow(obj->sigma*rr, 6);

  vcut = 4.0*obj->epsilon*(rs*rs - rs);
  dvcut = -24.0*rr*obj->epsilon*(2.0*rs*rs - rs);

  rr = 1.0/r;
  rs = pow(obj->sigma*rr, 6);

  *v = 4.0*obj->epsilon*(rs*rs - rs) - vcut - (r - obj->rc)*dvcut;
  *f = -(-24.0*rr*obj->epsilon*(2.0*rs*rs - rs) - dvcut);

  return 0;
}
