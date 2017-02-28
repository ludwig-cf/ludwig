/*****************************************************************************
 *
 *  pair_yukawa.c
 *
 *  This is a Yukawa potential:
 *
 *     V(r) = epsilon exp (-kappa r) / r
 *
 *  where r is a centre-centre separation. This is 'cut-and-shifted'
 *  to zero for both potential and force at cut-off rc.
 *
 *  This is a cut-and-shift version.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "pair_yukawa.h"

struct pair_yukawa_s {
  pe_t * pe;             /* Parallel environment */
  cs_t * cs;             /* Coordinate system */
  double epsilon;        /* Energy */
  double kappa;          /* Reciprocal length */
  double rc;             /* Cut off distance */
  double hminlocal;      /* Minimum gap (surface-surface) */
  double rminlocal;      /* Minimum separation (centre-centre) */
  double vlocal;         /* Contribution to potential */
};

/*****************************************************************************
 *
 *  pair_yukawa_create
 *
 *****************************************************************************/

int pair_yukawa_create(pe_t * pe, cs_t * cs, pair_yukawa_t ** pobj) {

  pair_yukawa_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (pair_yukawa_t *) calloc(1, sizeof(pair_yukawa_t));
  if (obj == NULL) pe_fatal(pe, "calloc(pair_yukawa_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_free
 *
 *****************************************************************************/

int pair_yukawa_free(pair_yukawa_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_info
 *
 *****************************************************************************/

int pair_yukawa_info(pair_yukawa_t * obj) {

  physics_t * phys;
  double kt;

  assert(obj);

  physics_ref(&phys);
  physics_kt(phys, &kt);

  pe_info(obj->pe, "\n");
  pe_info(obj->pe, "Yukawa potential\n");
  pe_info(obj->pe, "epsilon:                %14.7e\n", obj->epsilon);
  if (kt > 0.0) pe_info(obj->pe, "epsilon / kt            %14.7e\n", obj->epsilon/kt);
  pe_info(obj->pe, "kappa:                  %14.7e\n", obj->kappa);
  pe_info(obj->pe, "cut off (centre-centre) %14.7e\n", obj->rc);
  pe_info(obj->pe, "cut off / kappa         %14.7e\n", obj->rc/obj->kappa);

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_param_set
 *
 *****************************************************************************/

int pair_yukawa_param_set(pair_yukawa_t * obj, double epsilon, double kappa,
			  double rc) {

  assert(obj);

  obj->epsilon = epsilon;
  obj->kappa = kappa;
  obj->rc = rc;

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_register
 *
 *****************************************************************************/

int pair_yukawa_register(pair_yukawa_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_PAIR, obj, pair_yukawa_compute);
  interact_statistic_add(parent, INTERACT_PAIR, obj, pair_yukawa_stats);
  interact_rc_set(parent, INTERACT_PAIR, obj->rc);

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_compute
 *
 *****************************************************************************/

int pair_yukawa_compute(colloids_info_t * cinfo, void * self) {

  pair_yukawa_t * obj = (pair_yukawa_t *) self;

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];

  double r12[3];
  double f;
  double r, h, rr;
  double vcut;
  double dvcut;
  double ltot[3];

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(obj);

  cs_ltot(obj->cs, ltot);
  colloids_info_ncell(cinfo, ncell);

  vcut = obj->epsilon*exp(-obj->kappa*obj->rc)/obj->rc;
  dvcut = -vcut*(1.0/obj->rc + obj->kappa);

  obj->vlocal = 0.0;
  obj->rminlocal = ltot[X];
  obj->hminlocal = ltot[X];

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

                  cs_minimum_distance(obj->cs, pc1->s.r, pc2->s.r, r12);
		  r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);

		  if (r < obj->rminlocal) obj->rminlocal = r;
		  h = r - pc1->s.ah - pc2->s.ah;
		  if (h < obj->hminlocal) obj->hminlocal = h;
		  if (r >= obj->rc) continue;

		  rr = 1.0/r;
		  f = -(-obj->epsilon*exp(-obj->kappa*r)*rr*(rr + obj->kappa)
			- dvcut);

		  pc1->force[X] -= f*r12[X]*rr;
		  pc1->force[Y] -= f*r12[Y]*rr;
		  pc1->force[Z] -= f*r12[Z]*rr;
		  pc2->force[X] += f*r12[X]*rr;
		  pc2->force[Y] += f*r12[Y]*rr;
		  pc2->force[Z] += f*r12[Z]*rr;

		  obj->vlocal += obj->epsilon*exp(-obj->kappa*r)/r
		    - vcut - (r - obj->rc)*dvcut;
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
 *  pair_yukawa_stats
 *
 *****************************************************************************/

int pair_yukawa_stats(void * self, double * stat) {

  pair_yukawa_t * obj = (pair_yukawa_t *) self;

  assert(obj);
  assert(stat);

  stat[INTERACT_STAT_VLOCAL] = obj->vlocal;
  stat[INTERACT_STAT_HMINLOCAL] = obj->hminlocal;
  stat[INTERACT_STAT_RMINLOCAL] = obj->rminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_single
 *
 *  The cut-and-shift version is:
 *
 *    potential: V(r) - V(rc) - (r - rc) dV/dr|_rc
 *    force:     -(dV/dr - dV/dr|_rc) [\hat{r}_ij not included here]
 *
 *****************************************************************************/

int pair_yukawa_single(pair_yukawa_t * obj, double r, double * v, double * f) {

  double rr;      /* 1/r */
  double rrc;     /* 1/rc */
  double vcut;    /* V(rc) */
  double dvcut;   /* dV/dr|_rc */

  assert(obj);
  assert(r > 0.0);

  rr = 1.0/r;
  rrc = 1.0/obj->rc;

  vcut = obj->epsilon*exp(-obj->kappa*obj->rc)*rrc;
  dvcut = -vcut*(rrc + obj->kappa);

  *v = obj->epsilon*exp(-obj->kappa*r)/r - vcut - (r - obj->rc)*dvcut;
  *f = -(-obj->epsilon*exp(-obj->kappa*r)*rr*(rr + obj->kappa) - dvcut);

  return 0;
}
