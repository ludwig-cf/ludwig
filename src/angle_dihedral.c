//CHANGE1
/*****************************************************************************
 *
 *  angle_dihedral.c
 *
 *  Dihedral potential involving sites 0, 1, 2, and 3
 *
 *     m = (r_0 - r_1) x (r_2 - r_1)
 *     
 *     n = (r_2 - r_1) x (r_2 - r_3)
 *
 *     phi = sign(phi) arccos(m.n / |m| |n|)
 *
 *     sign(phi) = sign of [ (r_0 - r_1)).n ]
 *
 *     V(phi) = kappa * [1 + cos(mu * phi - phi0)]
 *
 *   where phi is the dihedral angle.
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
#include "angle_dihedral.h"

struct angle_dihedral_s {
  pe_t * pe;
  cs_t * cs;
  double kappa;
  int mu; 
  double phi0; 
  double vlocal;
  double phi_max; 
  double phi_min; 
};

/*****************************************************************************
 *
 *  angle_dihedral_create
 *
 *****************************************************************************/

int angle_dihedral_create(pe_t * pe, cs_t * cs, angle_dihedral_t ** pobj) {

  angle_dihedral_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (angle_dihedral_t *) calloc(1, sizeof(angle_dihedral_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(angle_dihedral) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  angle_dihedral_free
 *
 *****************************************************************************/

int angle_dihedral_free(angle_dihedral_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  angle_dihedral_param_set
 *
 *****************************************************************************/

int angle_dihedral_param_set(angle_dihedral_t * obj, double kappa, int mu, double phi0) {

  assert(obj);

  obj->kappa = kappa;
  obj->mu = mu;
  obj->phi0 = phi0;

  return 0;
}

/*****************************************************************************
 *
 *  angle_dihedral_info
 *
 *****************************************************************************/

int angle_dihedral_info(angle_dihedral_t * obj) {

  assert(obj);

  pe_info(obj->pe, "Bond angle\n");
  pe_info(obj->pe, "Type:                         dihedral\n");
  pe_info(obj->pe, "kappa:                       %14.7e\n", obj->kappa);
  pe_info(obj->pe, "mu:                           %d\n", obj->mu);
  pe_info(obj->pe, "phi0:                        %14.7e\n", obj->phi0);

  return 0;
}

/*****************************************************************************
 *
 *  angle_dihedral_register
 *
 *****************************************************************************/

int angle_dihedral_register(angle_dihedral_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_ANGLE_DIHEDRAL, obj, angle_dihedral_compute);
  interact_statistic_add(parent, INTERACT_ANGLE_DIHEDRAL, obj, angle_dihedral_stats);

  return 0;
}

/*****************************************************************************
 *
 *  angle_dihedral_compute
 *
 *****************************************************************************/

int angle_dihedral_compute(colloids_info_t * cinfo, void * self) {

  angle_dihedral_t * obj = (angle_dihedral_t *) self;

  double r0[3], r1[3], r2[3];  /* separations */
  double r0sq, r1sq, r2sq;     /* squared separations */
  double r0md, r1md, r2md;     /* moduli */
  double cosine;               /* of angle */
  double f0[3], f2[3];         /* forces */
  int b0, b1, b2;              /* index of the bonds */
  double m[3], n[3];           /* normal vectors */
  double msq, nsq;             /* squared value */
  double mmd, nmd;             /* moduli */
  double phi;                  /* dihedral angle */
  double dV_dphi;              /* dV / dphi */
  double v01, v12;             /* intermediate values used for calculation */ 
  double sign;                 /* sign of phi */

  colloid_t * pc = NULL;

  assert(obj);
  assert(cinfo);

  obj->vlocal = 0.0;
  obj->phi_min = +DBL_MAX;
  obj->phi_max = -DBL_MAX;
  colloids_info_local_head(cinfo, &pc);

  for (; pc; pc = pc->nextlocal) {

    assert(pc->s.nbonds);

    for (b0=0; b0<pc->s.nbonds; b0++) {

      assert(pc->bonded[b0]->s.index == pc->s.bond[b0]);
      assert(pc->bonded[b0]->s.nbonds);

      for (b1=0; b1<pc->bonded[b0]->s.nbonds; b1++) {

        assert(pc->bonded[b0]->s.bond[b1]);
        assert(pc->bonded[b0]->bonded[b1]);
        assert(pc->bonded[b0]->bonded[b1]->s.index == pc->bonded[b0]->s.bond[b1]);
      
        if (pc->bonded[b0]->bonded[b1]->s.index == pc->s.index) continue;

        assert(pc->bonded[b0]->bonded[b1]->s.nbonds);

        for (b2=0; b2<pc->bonded[b0]->bonded[b1]->s.nbonds; b2++) {
            
          assert(pc->bonded[b0]->bonded[b1]->bonded[b2]->s.index == pc->bonded[b0]->bonded[b1]->s.bond[b2]);

          if (pc->bonded[b0]->bonded[b1]->bonded[b2]->s.index == pc->bonded[b0]->s.index) continue;

          assert(pc->bonded[b0]->bonded[b1]->bonded[b2]->s.nbonds);

          if (pc->s.index < pc->bonded[b0]->bonded[b1]->bonded[b2]->s.index) continue; /* To avoid double counting */

          /* Bond between 0 and 1 */
          cs_minimum_distance(obj->cs, pc->bonded[b0]->s.r, pc->s.r, r0);
          r0sq = r0[X]*r0[X] + r0[Y]*r0[Y] + r0[Z]*r0[Z];
          r0md = sqrt(r0sq);

          /* Bond between 1 and 2 */
          cs_minimum_distance(obj->cs, pc->bonded[b0]->s.r, pc->bonded[b0]->bonded[b1]->s.r, r1);
          r1sq = r1[X]*r1[X] + r1[Y]*r1[Y] + r1[Z]*r1[Z];
          r1md = sqrt(r1sq);

          /* Bond between 2 and 3 */
          cs_minimum_distance(obj->cs, pc->bonded[b0]->bonded[b1]->bonded[b2]->s.r, pc->bonded[b0]->bonded[b1]->s.r, r2);
          r2sq = r2[X]*r2[X] + r2[Y]*r2[Y] + r2[Z]*r2[Z];
          r2md = sqrt(r2sq);

          /* calculate the normoal vector for surface 012 */
          cross_product(r0,r1,m);
          msq = m[X]*m[X] + m[Y]*m[Y] + m[Z]*m[Z];
          mmd = sqrt(msq);

          /* calculate the normoal vector for surface 123 */
          cross_product(r1,r2,n);
          nsq = n[X]*n[X] + n[Y]*n[Y] + n[Z]*n[Z];
          nmd = sqrt(nsq);

          cosine = dot_product(m, n) / (mmd*nmd);
          assert(cosine <= 1.0);
          assert(cosine >= -1.0);

          phi = acos(cosine);
          sign = copysign(1,dot_product(r0,n));
          phi *= sign;
        
          dV_dphi = obj->kappa * obj->mu * sin(obj->phi0 - obj->mu * phi);

          f0[X] = - dV_dphi * r1sq * m[X] / msq;
          f0[Y] = - dV_dphi * r1sq * m[Y] / msq;
          f0[Z] = - dV_dphi * r1sq * m[Z] / msq;

          f2[X] = dV_dphi * r1sq * n[X] / nsq;
          f2[Y] = dV_dphi * r1sq * n[Y] / nsq;
          f2[Z] = dV_dphi * r1sq * n[Z] / nsq;

          v01 = dot_product(r0,r1) / r1sq;
          v12 = dot_product(r1,r2) / r1sq;

          pc->force[X] += f0[X]; 
          pc->force[Y] += f0[Y];
          pc->force[Z] += f0[Z];

          pc->bonded[b0]->force[X] += -f0[X] + v01 * f0[X] - v12 * f2[X];   
          pc->bonded[b0]->force[Y] += -f0[Y] + v01 * f0[Y] - v12 * f2[Y]; 
          pc->bonded[b0]->force[Z] += -f0[Z] + v01 * f0[Z] - v12 * f2[Z]; 

          pc->bonded[b0]->bonded[b1]->force[X] += -f2[X] - v01 * f0[X] + v12 * f2[X];   
          pc->bonded[b0]->bonded[b1]->force[Y] += -f2[Y] - v01 * f0[Y] + v12 * f2[Y]; 
          pc->bonded[b0]->bonded[b1]->force[Z] += -f2[Z] - v01 * f0[Z] + v12 * f2[Z]; 

          pc->bonded[b0]->bonded[b1]->bonded[b2]->force[X] += f2[X]; 
          pc->bonded[b0]->bonded[b1]->bonded[b2]->force[Y] += f2[Y];
          pc->bonded[b0]->bonded[b1]->bonded[b2]->force[Z] += f2[Z];

          /* Potential energy */

          obj->vlocal += obj->kappa * (1 + cos(obj->mu * phi - obj->phi0));
          if (phi < obj->phi_min) obj->phi_min = phi;
          if (phi > obj->phi_max) obj->phi_max = phi;
        }
      }
    }
  }
  
  return 0;
}


/*****************************************************************************
 *
 *  angle_dihedral_stats
 *
 *****************************************************************************/

int angle_dihedral_stats(void * self, double * stats) {

  angle_dihedral_t * obj = (angle_dihedral_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL] = obj->vlocal;
  /* "rmax" "rmin" here are radians */
  stats[INTERACT_STAT_RMINLOCAL] = obj->phi_min;
  stats[INTERACT_STAT_RMAXLOCAL] = obj->phi_max;

  return 0;
}



