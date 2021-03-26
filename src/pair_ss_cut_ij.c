/*****************************************************************************
 *
 *  pair_ss_cut_ij.c
 *
 *  As per pair_ss_cut.c but different types (i,j) are allowed to
 *  have different parameters.
 *
 *  Note also here that the exponent nu is allowed to be a non-integer
 *  in the input.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kai Qi (kai.qi@epfl.ch)
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
#include "pair_ss_cut_ij.h"

/*****************************************************************************
 *
 *  pair_ss_cut_ij_create
 *
 *****************************************************************************/

int pair_ss_cut_ij_create(pe_t * pe, cs_t * cs, int ntypes, double * epsilon,
			  double * sigma, double * nu, double * hc,
			  pair_ss_cut_ij_t ** pobj) {

  pair_ss_cut_ij_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(ntypes > 0);
  assert(pobj);

  obj = (pair_ss_cut_ij_t *) calloc(1, sizeof(pair_ss_cut_ij_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(pair_ss_cut_ij_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->ntypes = ntypes;

  /* Matrices for the coefficients */

  obj->epsilon = (double **) calloc(ntypes, sizeof(double *));
  obj->sigma   = (double **) calloc(ntypes, sizeof(double *));
  obj->nu      = (double **) calloc(ntypes, sizeof(double *));
  obj->hc      = (double **) calloc(ntypes, sizeof(double *));
  assert(obj->epsilon);
  assert(obj->sigma);
  assert(obj->nu);
  assert(obj->hc);
  if (obj->epsilon == NULL) pe_fatal(pe, "calloc(pair_ss_cut_ij_t) failed\n");
  if (obj->sigma == NULL)   pe_fatal(pe, "calloc(pair_ss_cut_ij_t) failed\n");
  if (obj->nu == NULL)      pe_fatal(pe, "calloc(pair_ss_cut_ij_t) failed\n");
  if (obj->hc == NULL)      pe_fatal(pe, "calloc(pair_ss_cut_ij_t) failed\n");

  for (int i = 0; i < ntypes; i++) {
    obj->epsilon[i] = (double *) calloc(ntypes, sizeof(double));
    obj->sigma[i]   = (double *) calloc(ntypes, sizeof(double));
    obj->nu[i]      = (double *) calloc(ntypes, sizeof(double));
    obj->hc[i]      = (double *) calloc(ntypes, sizeof(double));
    assert(obj->epsilon[i]);
    assert(obj->sigma[i]);
    assert(obj->nu[i]);
    assert(obj->hc[i]);
    if (obj->epsilon[i] == NULL) pe_fatal(pe, "calloc(pair_ss_cut_ij_t)\n");
    if (obj->sigma[i] == NULL)   pe_fatal(pe, "calloc(pair_ss_cut_ij_t)\n");
    if (obj->nu[i] == NULL)      pe_fatal(pe, "calloc(pair_ss_cut_ij_t)\n");
    if (obj->hc[i] == NULL)      pe_fatal(pe, "calloc(pair_ss_cut_ij_t)\n");
  }

  pair_ss_cut_ij_param_set(obj, epsilon, sigma, nu, hc);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_free
 *
 *****************************************************************************/

int pair_ss_cut_ij_free(pair_ss_cut_ij_t * obj) {

  assert(obj);

  for (int ic = 0; ic < obj->ntypes; ic++) {
    free(obj->epsilon[ic]);
    free(obj->sigma[ic]);
    free(obj->nu[ic]);
    free(obj->hc[ic]);
  }
  free(obj->epsilon);
  free(obj->sigma);
  free(obj->nu);
  free(obj->hc);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_param_set
 *
 *  These quantities must form a symmetric matrix from the compressed
 *  n(n+1)/2 values. We expect the upper triangle, e.g., for three
 *  types: m_11 m_12 m_13 m_22 m_23 m_33.
 *
 *****************************************************************************/

int pair_ss_cut_ij_param_set(pair_ss_cut_ij_t * obj, double * epsilon,
			     double * sigma, double * nu, double * hc) {
  int index = 0;

  assert(obj);
  assert(epsilon);
  assert(sigma);
  assert(nu);
  assert(hc);

  /* Upper triangle */

  for (int i = 0; i < obj->ntypes; i++) {
    for (int j = i; j < obj->ntypes; j++) {
      obj->epsilon[i][j] = epsilon[index];
      obj->sigma[i][j]   = sigma[index];
      obj->nu[i][j]      = nu[index];
      obj->hc[i][j]      = hc[index];
      index += 1;
    }
  }

  /* Lower triangle */

  for (int i = 0; i < obj->ntypes; i++) {
    for (int j = 0; j < i; j++) {
      obj->epsilon[i][j] = obj->epsilon[j][i];
      obj->sigma[i][j]   = obj->sigma[j][i];
      obj->nu[i][j]      = obj->nu[j][i];
      obj->hc[i][j]      = obj->hc[j][i];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_info
 *
 *****************************************************************************/

int pair_ss_cut_ij_info(pair_ss_cut_ij_t * obj) {

  physics_t * phys;
  double kt;

  physics_ref(&phys);
  physics_kt(phys, &kt);

  pe_info(obj->pe, "\n");
  pe_info(obj->pe, "Soft sphere potential (pair types ij)\n");

  for (int i = 0; i < obj->ntypes; i++) {
    for (int j = i; j < obj->ntypes; j++) {

      pe_info(obj->pe, "Pair type:    %d %d\n", i, j);
      pe_info(obj->pe, "epsilon:      %14.7e\n", obj->epsilon[i][j]);
      pe_info(obj->pe, "sigma:        %14.7e\n", obj->sigma[i][j]);
      pe_info(obj->pe, "exponent nu:  %14.7e\n", obj->nu[i][j]);
      pe_info(obj->pe, "cut off:      %14.7e\n", obj->hc[i][j]);

      if (kt > 0.0) {
	pe_info(obj->pe, "epsilon / kT: %14.7e\n", obj->epsilon[i][j]/kt);
      }
      pe_info(obj->pe, "\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_register
 *
 *****************************************************************************/

int pair_ss_cut_ij_register(pair_ss_cut_ij_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  double hcmax = 0.0;

  interact_potential_add(parent, INTERACT_PAIR, obj, pair_ss_cut_ij_compute);
  interact_statistic_add(parent, INTERACT_PAIR, obj, pair_ss_cut_ij_stats);

  for (int i = 0; i < obj->ntypes; i++) {
    for (int j = 0; j < obj->ntypes; j++) {
      hcmax = dmax(hcmax, obj->hc[i][j]);
    }
  }
  interact_hc_set(parent, INTERACT_PAIR, hcmax);

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_compute
 *
 *****************************************************************************/

int pair_ss_cut_ij_compute(colloids_info_t * cinfo, void * obj) {

  pair_ss_cut_ij_t * self = (pair_ss_cut_ij_t *) obj;

  int ic1, jc1, kc1, ic2, jc2, kc2;
  int di[2], dj[2], dk[2];
  int ncell[3];
  int it1, it2;

  double r;                             /* centre-centre sepration */
  double h;                             /* surface-surface separation */
  double rh;                            /* reciprocal h */
  double r12[3];                        /* centre-centre min distance 1->2 */
  double f;
  double ltot[3];
  double epsilon, sigma, nu, hc;

  double rsigma[self->ntypes][self->ntypes]; /* reciproal sigma */
  double vcut[self->ntypes][self->ntypes];   /* potential at cut off */
  double dvcut[self->ntypes][self->ntypes];  /* derivative at cut off */
  colloid_t * pc1;
  colloid_t * pc2;

  assert(cinfo);
  assert(self);

  cs_ltot(self->cs, ltot);

  self->vlocal = 0.0;
  self->hminlocal = dmax(ltot[X], dmax(ltot[Y], ltot[Z]));
  self->rminlocal = self->hminlocal;

  for (int i = 0; i < self->ntypes; i++) {
    for (int j = 0; j < self->ntypes; j++) {
      epsilon = self->epsilon[i][j];
      sigma   = self->sigma[i][j];
      nu      = self->nu[i][j];
      hc      = self->hc[i][j];
      rsigma[i][j] = 1.0/sigma;
      vcut[i][j] = epsilon*pow(sigma/self->hc[i][j], nu);
      dvcut[i][j] = -epsilon*nu*rsigma[i][j]*pow(sigma/hc, nu + 1.0);
    }
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

                  it1 = pc1->s.inter_type;
                  it2 = pc2->s.inter_type;
		  assert(it1 < self->ntypes);
		  assert(it2 < self->ntypes);

		  if (h > self->hc[it1][it2]) continue;
		  assert(h > 0.0);

		  rh = 1.0/h;

		  epsilon = self->epsilon[it1][it2];
		  sigma   = self->sigma[it1][it2];
		  nu      = self->nu[it1][it2];
		  hc      = self->hc[it1][it2];

		  self->vlocal += epsilon*pow(rh*sigma, nu)
		    - vcut[it1][it2] - (h - hc)*dvcut[it1][it2];
		  f = -(-epsilon*nu*rsigma[it1][it2]
			*pow(rh*sigma, nu + 1.0) - dvcut[it1][it2]);

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
 *  pair_ss_cut_ij_stats
 *
 *****************************************************************************/

int pair_ss_cut_ij_stats(void * obj, double * stat) {

  pair_ss_cut_ij_t * self = (pair_ss_cut_ij_t *) obj;

  assert(self);

  stat[INTERACT_STAT_VLOCAL]    = self->vlocal;
  stat[INTERACT_STAT_HMINLOCAL] = self->hminlocal;
  stat[INTERACT_STAT_RMINLOCAL] = self->rminlocal;

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_single
 *
 *  Returns scalar part of force for given h, and potential.
 *
 *****************************************************************************/

int pair_ss_cut_ij_single(pair_ss_cut_ij_t * obj, int i, int j, double h,
			  double * f, double * v) {
  double rh;
  double rsigma;
  double vcut, dvcut;

  double epsilon, sigma, nu, hc;

  assert(obj);
  assert(0 <= i && i < obj->ntypes);
  assert(0 <= j && j < obj->ntypes);
  assert(h > 0.0);

  epsilon = obj->epsilon[i][j];
  sigma   = obj->sigma[i][j];
  nu      = obj->nu[i][j];
  hc      = obj->hc[i][j];

  rh = 1.0/h;
  rsigma = 1.0/sigma;
  vcut = epsilon*pow(sigma/hc, nu);
  dvcut = -epsilon*nu*rsigma*pow(sigma/hc, nu + 1.0);

  *v = epsilon*pow(rh*sigma, nu) - vcut - (h - hc)*dvcut;
  *f = -(-epsilon*nu*rsigma*pow(rh*sigma, nu + 1.0) - dvcut);

  /* Remember 'f' is -(dv/dr - dv/dr|_rc) */

  return 0;
}
