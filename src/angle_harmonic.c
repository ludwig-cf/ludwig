//CHANGE1
/*****************************************************************************
 *
 *  angle_harmonic.c
 *
 *  Bending potential involving sites 1, 2, and 3
 *
 *     cos(theta) = (r_1 - r_2).(r_3 - r_2) / | r_1 - r_2 | | r_3 - r_2 |
 *
 *     V(2) = 0.5 * kappa * (theta - theta0)^2
 *
 *   where theta is the angle at 2.
 *
 *   Edinburgh Soft Matter and Statistical Physics Group and
 *   Edinburgh Parallel Computing Centre
 *
 *   (c) 2014-2017 The University of Edinburgh
 *
 *   Contributing authors:
 *   Kevin Stratford (kevin@epcc.ed.ac.uk)
 *   Kai Qi (kai.qi@epfl.ch)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "colloids.h"
#include "angle_harmonic.h"

struct angle_harmonic_s {
  pe_t * pe;
  cs_t * cs;
  double kappa;
  double theta0; 
  double vlocal;
  double theta_max;
  double theta_min;
};

/*****************************************************************************
 *
 *  angle_harmonic_create
 *
 *****************************************************************************/

int angle_harmonic_create(pe_t * pe, cs_t * cs, angle_harmonic_t ** pobj) {

  angle_harmonic_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (angle_harmonic_t *) calloc(1, sizeof(angle_harmonic_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(angle_harmonic) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_free
 *
 *****************************************************************************/

int angle_harmonic_free(angle_harmonic_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_param_set
 *
 *****************************************************************************/

int angle_harmonic_param_set(angle_harmonic_t * obj, double kappa, double theta0) {

  assert(obj);

  obj->kappa = kappa;
  obj->theta0 = theta0;

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_info
 *
 *****************************************************************************/

int angle_harmonic_info(angle_harmonic_t * obj) {

  assert(obj);

  pe_info(obj->pe, "Bond angle\n");
  pe_info(obj->pe, "Type:                          harmonic\n");
  pe_info(obj->pe, "kappa:                        %14.7e\n", obj->kappa);
  pe_info(obj->pe, "theta0:                       %14.7e\n", obj->theta0);

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_register
 *
 *****************************************************************************/

int angle_harmonic_register(angle_harmonic_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_ANGLE_HARMONIC, obj, angle_harmonic_compute);
  interact_statistic_add(parent, INTERACT_ANGLE_HARMONIC, obj, angle_harmonic_stats);

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_compute
 *
 *****************************************************************************/

int angle_harmonic_compute(colloids_info_t * cinfo, void * self) {

  angle_harmonic_t * obj = (angle_harmonic_t *) self;

  double r0[3], r1[3];       /* separations */
  double r0sq, r1sq;         /* squared separations */
  double r0md, r1md;         /* moduli */
  double cosine;             /* of angle */
  double v0, v1, v01;        /* potential coefficients */
  double f0[3], f1[3];       /* forces */
  double theta;              /* angle */
  double dtheta;             /* angle difference */
  double sine;               /* of angle */
  int b0, b1;                /* index of the bonds */


  colloid_t * pc = NULL;

  assert(obj);
  assert(cinfo);

  obj->vlocal = 0.0;
  obj->theta_min = +DBL_MAX;
  obj->theta_max = -DBL_MAX;
  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {
    if (pc->s.nangles == 0) continue;

    assert(pc->s.nbonds>1);

    for (b0=0; b0<pc->s.nbonds-1; b0++)
      for (b1=b0+1; b1<pc->s.nbonds; b1++) {

        assert(pc->bonded[b0]);
        assert(pc->bonded[b1]);
        assert(pc->bonded[b0]->s.index == pc->s.bond[b0]);
        assert(pc->bonded[b1]->s.index == pc->s.bond[b1]);

        /* Bond b0 is pc -> bonded[b0] */

        cs_minimum_distance(obj->cs, pc->s.r, pc->bonded[b0]->s.r, r0);
        r0sq = r0[X]*r0[X] + r0[Y]*r0[Y] + r0[Z]*r0[Z];
        r0md = sqrt(r0sq);

        /* Bond b1 is pc -> bonded[b1] */

        cs_minimum_distance(obj->cs, pc->s.r, pc->bonded[b1]->s.r, r1);
        r1sq = r1[X]*r1[X] + r1[Y]*r1[Y] + r1[Z]*r1[Z];
        r1md = sqrt(r1sq);

        cosine = dot_product(r0, r1) / (r0md*r1md);
        assert(cosine <= 1.0);
        assert(cosine >= -1.0);

        v0  = obj->kappa*cosine/r0sq;
        v01 = obj->kappa/(r0md*r1md);
        v1  = obj->kappa*cosine/r1sq;

        theta = acos(cosine);
        dtheta = theta-obj->theta0;
        sine = 1-sqrt(cosine);

        if(sine>1e-5) {
            f0[X] = dtheta/sine*(v01*r1[X] - v0*r0[X]);
            f0[Y] = dtheta/sine*(v01*r1[Y] - v0*r0[Y]);
            f0[Z] = dtheta/sine*(v01*r1[Z] - v0*r0[Z]);
                                            
            f1[X] = dtheta/sine*(v01*r0[X] - v1*r1[X]);
            f1[Y] = dtheta/sine*(v01*r0[Y] - v1*r1[Y]);
            f1[Z] = dtheta/sine*(v01*r0[Z] - v1*r1[Z]);
        }
        else {
            /* At the singularity point, force is not defined. Therefore, set it to an unstable equilibrium value.*/
            f0[X] = 0;
            f0[Y] = 0;
            f0[Z] = 0;
            f1[X] = 0;
            f1[Y] = 0;
            f1[Z] = 0;
        }

        /* Accumulate forces */

        pc->bonded[b0]->force[X] += f0[X];
        pc->bonded[b0]->force[Y] += f0[Y];
        pc->bonded[b0]->force[Z] += f0[Z];

        pc->force[X] -= (f0[X] + f1[X]);
        pc->force[Y] -= (f0[Y] + f1[Y]);
        pc->force[Z] -= (f0[Z] + f1[Z]);

        pc->bonded[b1]->force[X] += f1[X];
        pc->bonded[b1]->force[Y] += f1[Y];
        pc->bonded[b1]->force[Z] += f1[Z];

        /* Potential energy */

        obj->vlocal += 0.5*obj->kappa*dtheta*dtheta;
        if (theta < obj->theta_min) obj->theta_min = theta;
        if (theta > obj->theta_max) obj->theta_max = theta;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  angle_harmonic_stats
 *
 *****************************************************************************/

int angle_harmonic_stats(void * self, double * stats) {

  angle_harmonic_t * obj = (angle_harmonic_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL] = obj->vlocal;
  /* "rmax" "rmin" here are radians */
  stats[INTERACT_STAT_RMINLOCAL] = obj->theta_min;
  stats[INTERACT_STAT_RMAXLOCAL] = obj->theta_max;

  return 0;
}

