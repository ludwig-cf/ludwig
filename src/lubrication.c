/*****************************************************************************
 *
 *  lubrication.c
 *
 *  Colloid-colloid lubrication corrections for particles close
 *  to contact.
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
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "colloids.h"
#include "lubrication.h"

struct lubrication_s {
  pe_t * pe;
  cs_t * cs;
  double hminlocal;
  double rch[LUBRICATION_SS_MAX];    /* Cut offs for different components */
  double rrch[LUBRICATION_SS_MAX];   /* Table of reciprocal cut offs */
  double rchmax;
};

/*****************************************************************************
 *
 *  lubrication_create
 *
 *****************************************************************************/

int lubrication_create(pe_t * pe, cs_t * cs, lubr_t ** pobj) {

  lubr_t * obj = NULL;

  assert(pe);
  assert(pobj);

  obj = (lubr_t *) calloc(1, sizeof(lubr_t));
  if (obj == NULL) pe_fatal(pe, "calloc(lubr_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_free
 *
 *****************************************************************************/

int lubrication_free(lubr_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_compute
 *
 *  Call back function for computing sphere-sphere corrections.
 *
 *****************************************************************************/

int lubrication_compute(colloids_info_t * cinfo, void * self) {

  lubr_t * obj = (lubr_t *) self;

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];

  double ran[2];  /* Random numbers for fluctuation dissipation correction */
  double r12[3];
  double f[3];
  double ltot[3];

  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(obj);

  cs_ltot(obj->cs, ltot);

  obj->hminlocal = ltot[X];
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

                  cs_minimum_distance(obj->cs, pc1->s.r, pc2->s.r, r12);
		  util_ranlcg_reap_gaussian(&pc1->s.rng, ran);

		  lubrication_single(obj, pc1->s.ah, pc2->s.ah, pc1->s.v,
				     pc2->s.v, r12, ran, f);

		  pc1->force[X] += f[X];
		  pc1->force[Y] += f[Y];
		  pc1->force[Z] += f[Z];

		  pc2->force[X] -= f[X];
		  pc2->force[Y] -= f[Y];
		  pc2->force[Z] -= f[Z];

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
 *  lubrication_stats
 *
 *****************************************************************************/

int lubrication_stats(void * obj, double * stats) {

  lubr_t * self = (lubr_t *) obj;

  assert(self);

  stats[INTERACT_STAT_HMINLOCAL] = self->hminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_rch_set
 *
 *****************************************************************************/

int lubrication_rch_set(lubr_t * obj, lubr_ss_enum_t type, double rc) {

  assert(obj);
  assert(type < LUBRICATION_SS_MAX);
  assert(rc >= 0.0);

  if (rc < 0.0) rc = 0.0;

  obj->rch[type] = rc;
  if (rc > 0.0) obj->rrch[type] = 1.0/rc;
  obj->rchmax = dmax(obj->rchmax, rc);

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_register
 *
 *****************************************************************************/

int lubrication_register(lubr_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_LUBR, obj, lubrication_compute);
  interact_statistic_add(parent, INTERACT_LUBR, obj, lubrication_stats);
  interact_hc_set(parent, INTERACT_LUBR, obj->rchmax);

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_rchmax
 *
 *****************************************************************************/

int lubrication_rchmax(lubr_t * obj, double * rchmax) {

  assert(obj);

  *rchmax = obj->rchmax;

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_single
 *
 *  Compute the net lubrication correction for the two colloids
 *  which are separated by vector r_ij (the size of which is h
 *  i.e., centre-centre distance).
 *
 *  If noise is on, then an additional random force is required to
 *  satisfy the fluctuation-dissipation theorem. The size of the
 *  random component depends on the "temperature", and is just
 *  added to the lubrication contribution.
 *
 *****************************************************************************/

int lubrication_single(lubr_t * lubr, double a1, double a2,
		       const double u1[3], const double u2[3],
		       const double r12[3], const double ran[2], double f[3]) {
  int ia;
  double h;        /* Separation */
  double hr;       /* Reduced separation */
  double eta;      /* viscosity */
  double fmod;
  double rh, rhr, rrc;
  double rdotdu;
  double rhat[3];
  double kt;
  physics_t * phys = NULL;
  PI_DOUBLE(pi);

  assert(lubr);
  physics_ref(&phys);

  for (ia = 0; ia < 3; ia++) {
    f[ia] = 0.0;
  }

  h = modulus(r12);
  hr = h - a1 - a2;
  if (hr < lubr->hminlocal) lubr->hminlocal = hr;

  if (hr < lubr->rch[LUBRICATION_SS_FNORM]) {

    physics_kt(phys, &kt);
    physics_eta_shear(phys, &eta);

    rhr = 1.0/hr;
    rrc  = lubr->rrch[LUBRICATION_SS_FNORM];
    fmod = -6.0*pi*eta*a1*a1*a2*a2*(rhr - rrc)/((a1 + a1)*(a2 + a2));

    /* Fluctuation/dissipation contribution */
    fmod += ran[0]*sqrt(-2.0*kt*fmod);

    rh = 1.0/h;
    rdotdu = 0.0;

    for (ia = 0; ia < 3; ia++) {
      rhat[ia] = rh*r12[ia];
      rdotdu += rhat[ia]*(u1[ia] - u2[ia]);
    }

    for (ia = 0; ia < 3; ia++) {
      f[ia] += fmod*rdotdu*rhat[ia];
    }
  }

  /* Tangential lubrication correction */

  if (hr < lubr->rch[LUBRICATION_SS_FTANG]) {

    physics_kt(phys, &kt);
    physics_eta_shear(phys, &eta);

    rhr = 1.0/hr;
    rh  = 0.5*(a1 + a2)*rhr;
    rrc = 0.5*(a1 + a2)*lubr->rrch[LUBRICATION_SS_FTANG];

    fmod = -(24.0/15.0)*pi*eta*a1*a2*(2.0*a1*a1 + a1*a2 + 2.0*a2*a2)
      *(log(rh) - log(rrc)) / ((a1+a2)*(a1+a2)*(a1+a2));

    fmod += ran[1]*sqrt(-2.0*kt*fmod);

    rh = 1.0/h;
    rdotdu = 0.0;

    for (ia = 0; ia < 3; ia++) {
      rhat[ia] = rh*r12[ia];
      rdotdu += rhat[ia]*(u1[ia] - u2[ia]);
    }

    for (ia = 0; ia < 3; ia++) {
      f[ia] += fmod*((u1[ia] - u2[ia]) - rdotdu*rhat[ia]);
    }
  }

  return 0;
}
