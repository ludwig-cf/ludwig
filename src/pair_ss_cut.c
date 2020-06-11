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
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "pair_ss_cut.h"

struct pair_ss_cut_s {
  pe_t * pe;             /* Parallel environemnt */
  cs_t * cs;             /* Coordinate system */
  double epsilon[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];        /* epsilon (energy) */
  double sigma[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];          /* sigma (length) */
  double nu[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];             /* exponent */
  double hc[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];             /* cut-off */
  double vlocal;         /* local contribution to energy */
  double hminlocal;      /* local nearest separation */
  double rminlocal;      /* local min centre-centre separation */
};

/*****************************************************************************
 *
 *  pair_ss_cut_create
 *
 *****************************************************************************/

int pair_ss_cut_create(pe_t * pe, cs_t * cs, pair_ss_cut_t ** pobj) {

  pair_ss_cut_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (pair_ss_cut_t *) calloc(1, sizeof(pair_ss_cut_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(pair_ss_cut_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_free
 *
 *****************************************************************************/

int pair_ss_cut_free(pair_ss_cut_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_param_set
 *
 *****************************************************************************/

int pair_ss_cut_param_set(pair_ss_cut_t * obj, double epsilon[][NUM_INT_PART_TYPES], double sigma[][NUM_INT_PART_TYPES],
			  double nu[][NUM_INT_PART_TYPES], double hc[][NUM_INT_PART_TYPES]) {
  assert(obj);

  for (int i=0;i<NUM_INT_PART_TYPES;i++)
    for (int j=0;j<NUM_INT_PART_TYPES;j++) {
        obj->epsilon[i][j] = epsilon[i][j];
        obj->sigma[i][j] = sigma[i][j];
        obj->nu[i][j] = nu[i][j];
        obj->hc[i][j] = hc[i][j];
    }

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
  physics_kt(phys, &kt);

  pe_info(obj->pe, "\n");
  pe_info(obj->pe, "Soft sphere potential\n");

  
  pe_info(obj->pe, "epsilon:                  ");
  for(int i=0;i<NUM_INT_PART_TYPES;i++) {
    for(int j=0;j<NUM_INT_PART_TYPES;j++) 
        pe_info(obj->pe, "%14.7e  ", obj->epsilon[i][j]);
    pe_info(obj->pe, "\n                          ");
  }
  pe_info(obj->pe, "\n");

  pe_info(obj->pe, "sigma:                    ");
  for(int i=0;i<NUM_INT_PART_TYPES;i++) {
    for(int j=0;j<NUM_INT_PART_TYPES;j++) 
        pe_info(obj->pe, "%14.7e  ", obj->sigma[i][j]);
    pe_info(obj->pe, "\n                          ");
  }
  pe_info(obj->pe, "\n");

  pe_info(obj->pe, "exponent nu:              ");
  for(int i=0;i<NUM_INT_PART_TYPES;i++) {
    for(int j=0;j<NUM_INT_PART_TYPES;j++) 
        pe_info(obj->pe, "%14.7e  ", obj->nu[i][j]);
    pe_info(obj->pe, "\n                          ");
  }
  pe_info(obj->pe, "\n");

  pe_info(obj->pe, "cut off (surface-surface) ");
  for(int i=0;i<NUM_INT_PART_TYPES;i++) {
    for(int j=0;j<NUM_INT_PART_TYPES;j++) 
        pe_info(obj->pe, "%14.7e  ", obj->hc[i][j]);
    pe_info(obj->pe, "\n                          ");
  }
  pe_info(obj->pe, "\n");

  if (kt > 0.0) {
    pe_info(obj->pe, "epsilon / kT              ");
    for(int i=0;i<NUM_INT_PART_TYPES;i++) {
      for(int j=0;j<NUM_INT_PART_TYPES;j++) 
          pe_info(obj->pe, "%14.7e  ", obj->epsilon[i][j]/kt);
      pe_info(obj->pe, "\n                          ");
    }
    pe_info(obj->pe, "\n");
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

  double hcmax=0;
  for(int i=0;i<NUM_INT_PART_TYPES;i++)
    for(int j=0;j<NUM_INT_PART_TYPES;j++)
        hcmax=dmax(hcmax,obj->hc[i][j]);
  interact_hc_set(parent, INTERACT_PAIR, hcmax);

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
  double rsigma[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES]; /* reciproal sigma */
  double vcut[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];   /* potential at cut off */
  double dvcut[NUM_INT_PART_TYPES][NUM_INT_PART_TYPES];  /* derivative at cut off */
  double r12[3];                        /* centre-centre min distance 1->2 */
  double f;
  double ltot[3];
  int it1,it2;

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(self);

  cs_ltot(self->cs, ltot);

  self->vlocal = 0.0;
  self->hminlocal = dmax(ltot[X], dmax(ltot[Y], ltot[Z]));
  self->rminlocal = self->hminlocal;

  for(int i=0;i<NUM_INT_PART_TYPES;i++)
    for(int j=0;j<NUM_INT_PART_TYPES;j++) {
        rsigma[i][j] = 1.0/self->sigma[i][j];
        vcut[i][j] = self->epsilon[i][j]*pow(self->sigma[i][j]/self->hc[i][j], self->nu[i][j]);
        dvcut[i][j] = -self->epsilon[i][j]*self->nu[i][j]*rsigma[i][j]*pow(self->sigma[i][j]/self->hc[i][j], self->nu[i][j]+1);
  }

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

		  cs_minimum_distance(self->cs, pc1->s.r, pc2->s.r, r12);
		  r = sqrt(r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z]);
		  if (r < self->rminlocal) self->rminlocal = r;

		  h = r - pc1->s.ah - pc2->s.ah;
		  if (h < self->hminlocal) self->hminlocal = h;

                  it1=pc1->s.inter_type;
                  it2=pc2->s.inter_type;

		  if (h > self->hc[it1][it2]) continue;
		  assert(h > 0.0);

		  rh = 1.0/h;

		  self->vlocal += self->epsilon[it1][it2]*pow(rh*self->sigma[it1][it2], self->nu[it1][it2])
		    - vcut[it1][it2] - (h - self->hc[it1][it2])*dvcut[it1][it2];
		  f = -(-self->epsilon[it1][it2]*self->nu[it1][it2]*rsigma[it1][it2]
			*pow(rh*self->sigma[it1][it2], self->nu[it1][it2]+1) - dvcut[it1][it2]);

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

int pair_ss_cut_single(pair_ss_cut_t * obj, double h, double * f, double * v,int it1, int it2) {

  double rh;
  double rsigma;
  double vcut, dvcut;

  assert(obj);
  assert(h > 0.0);

  rh = 1.0/h;
  rsigma = 1.0/obj->sigma[it1][it2];
  vcut = obj->epsilon[it1][it2]*pow(obj->sigma[it1][it2]/obj->hc[it1][it2], obj->nu[it1][it2]);
  dvcut = -obj->epsilon[it1][it2]*obj->nu[it1][it2]*rsigma*pow(obj->sigma[it1][it2]/obj->hc[it1][it2], obj->nu[it1][it2]+1);

  *v = obj->epsilon[it1][it2]*pow(rh*obj->sigma[it1][it2], obj->nu[it1][it2]) - vcut - (h - obj->hc[it1][it2])*dvcut;
  *f = -(-obj->epsilon[it1][it2]*obj->nu[it1][it2]*rsigma*pow(rh*obj->sigma[it1][it2], obj->nu[it1][it2]+1) - dvcut);

  /* Remember 'f' is -(dv/dr - dv/dr|_rc) */

  return 0;
}
