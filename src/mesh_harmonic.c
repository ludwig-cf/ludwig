//CHANGE1
/*****************************************************************************
 *
 *  mesh_harmonic.c
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "mesh_harmonic.h"

struct mesh_harmonic_s {
  pe_t * pe;           /* Parallel environment */
  cs_t * cs;           /* Coordinate system */
  double k;            /* spring constant */
  double r0;            /* spring constant */

  double vlocal;       /* Bond potential contribution */
  double rminlocal;    /* Minimum bond extent */
  double rmaxlocal;    /* Maximum bond extension */
  double bondlocal;    /* Number of bonds computed (double) */
};

/*****************************************************************************
 *
 *  mesh_harmonic_create
 *
 *****************************************************************************/

int mesh_harmonic_create(pe_t * pe, cs_t * cs, mesh_harmonic_t ** pobj) {

  mesh_harmonic_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (mesh_harmonic_t *) calloc(1, sizeof(mesh_harmonic_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(mesh_harmonic_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_free
 *
 *****************************************************************************/

int mesh_harmonic_free(mesh_harmonic_t * obj) {

  assert(obj);

  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_param_set
 *
 *****************************************************************************/

int mesh_harmonic_param_set(mesh_harmonic_t * obj, double k, double r0) {

  assert(obj);

  obj->k = k;
  obj->r0 = r0;

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_info
 *
 *****************************************************************************/

int mesh_harmonic_info(mesh_harmonic_t * obj) {

  assert(obj);

  pe_info(obj->pe, "Harmonic mesh\n");
  pe_info(obj->pe, "Spring constant:             %14.7e\n", obj->k);
  pe_info(obj->pe, "Equilibrium separation:      %14.7e\n", obj->r0);

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_register
 *
 *****************************************************************************/

int mesh_harmonic_register(mesh_harmonic_t * obj, interact_t * parent) {

  assert(obj);
  assert(parent);

  interact_potential_add(parent, INTERACT_MESH_HARMONIC, obj, mesh_harmonic_compute);
  interact_statistic_add(parent, INTERACT_MESH_HARMONIC, obj, mesh_harmonic_stats);

/* -----> CHEMOVESICLE V2 */
/* Vesicle size constraint on cell width is now done by colloids_rt.c */
  interact_rc_set(parent, INTERACT_MESH_HARMONIC, obj->r0);
/* <----- */

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_compute
 *
 *****************************************************************************/

int mesh_harmonic_compute(colloids_info_t * cinfo, void * self) {

  mesh_harmonic_t * obj = (mesh_harmonic_t *) self;

  int n;
  double r12[3], dcentre[3];
  double r2min, r2max;
  double r2,f;
  double r2md;
  double r0;

  colloid_t * pc = NULL;
  assert(cinfo);
  assert(obj);

  colloids_info_local_head(cinfo, &pc);

  obj->vlocal = 0;
  obj->bondlocal = 0.0;
  int tot = 0;
  for (; pc; pc = pc->nextlocal) {

    if (pc->s.nbonds_mesh == 0) pe_fatal(cinfo->pe, "One colloid has nbonds_mesh = 0\n");;

    for (n = 0; n < pc->s.nbonds_mesh; n++) {
      assert(pc->bonded_mesh[n]);
      
      if (pc->s.index == 1) continue; 
      /*Force on central particle is calculated implicity by summing together the counterforce of each edge particle */

      if (pc->s.index > pc->bonded_mesh[n]->s.index && pc->bonded_mesh[n]->s.index != 1) continue;
      /* Compute force arising on each particle from single bond */

      /* Retrieve equilibrium distance */
      for (int index = 0; index < 7; index++) {
        if (pc->s.tuple.indices[index] == pc->bonded_mesh[n]->s.index) {

          r0 = pc->s.tuple.r0s[index];
        }
      }

      obj->r0 = r0;

      cs_minimum_distance(obj->cs, pc->s.r, pc->bonded_mesh[n]->s.r, r12);
      r2 = r12[X]*r12[X] + r12[Y]*r12[Y] + r12[Z]*r12[Z];
      r2md = sqrt(r2);


/* LIGHTHOUSE
      r2min = 4*obj->r0*obj->r0;
      r2max = 0.0;
      printf("r2min = %14.7e, r2max = %14.7e, r2 = %14.7e\n", r2min, r2max, r2);
      if (r2 < r2min) r2min = r2;
      if (r2 > r2max) r2max = r2;
      if (r2 > 4*obj->r0*obj->r0) pe_fatal(obj->pe, "Broken harmonic bond\n");
*/
      obj->vlocal += 0.5*obj->k*(r2md-obj->r0)*(r2md-obj->r0);
      obj->bondlocal += 1.0;
      f = -obj->k*(r2md-obj->r0)/r2md;

      pc->force[X] -= f*r12[X];
      pc->force[Y] -= f*r12[Y];
      pc->force[Z] -= f*r12[Z];

      pc->bonded_mesh[n]->force[X] += f*r12[X];
      pc->bonded_mesh[n]->force[Y] += f*r12[Y];
      pc->bonded_mesh[n]->force[Z] += f*r12[Z];

/* For visualization purposes */
// Increment force/torque on the state attribute
      pc->s.fsprings[X] -= f*r12[X];
      pc->s.fsprings[Y] -= f*r12[Y];
      pc->s.fsprings[Z] -= f*r12[Z];

      cs_minimum_distance(obj->cs, pc->s.r, pc->centerofmass, dcentre);
      pc->s.tsprings[X] -= dcentre[Y]*f*r12[Z] - dcentre[Z]*f*r12[Y];
      pc->s.tsprings[Y] -= dcentre[Z]*f*r12[X] - dcentre[X]*f*r12[Z];
      pc->s.tsprings[Z] -= dcentre[X]*f*r12[Y] - dcentre[Y]*f*r12[X];

// Increment reciprocal force/torque on the state attribute
      pc->bonded_mesh[n]->s.fsprings[X] += f*r12[X];
      pc->bonded_mesh[n]->s.fsprings[Y] += f*r12[Y];
      pc->bonded_mesh[n]->s.fsprings[Z] += f*r12[Z];

// Careful here, the torque has to be calculated with the position of the linked bead
      cs_minimum_distance(obj->cs, pc->bonded_mesh[n]->s.r, pc->centerofmass, dcentre);
      pc->bonded_mesh[n]->s.tsprings[X] += dcentre[Y]*f*r12[Z] - dcentre[Z]*f*r12[Y];
      pc->bonded_mesh[n]->s.tsprings[Y] += dcentre[Z]*f*r12[X] - dcentre[X]*f*r12[Z];
      pc->bonded_mesh[n]->s.tsprings[Z] += dcentre[X]*f*r12[Y] - dcentre[Y]*f*r12[X];

    }
  }

  obj->rminlocal = sqrt(r2min);
  obj->rmaxlocal = sqrt(r2max);

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_stats
 *
 *****************************************************************************/

int mesh_harmonic_stats(void * self, double * stats) {

  mesh_harmonic_t * obj = (mesh_harmonic_t *) self;

  assert(obj);
  assert(stats);

  stats[INTERACT_STAT_VLOCAL]    = obj->vlocal;
  stats[INTERACT_STAT_RMINLOCAL] = obj->rminlocal;
  stats[INTERACT_STAT_RMAXLOCAL] = obj->rmaxlocal;

  return 0;
}

/*****************************************************************************
 *
 *  mesh_harmonic_single
 *
 *  For a single mesh, compute v and |f| given r.
 *
 *****************************************************************************/

int mesh_harmonic_single(mesh_harmonic_t * obj, double r, double * v, double * f) {

  assert(obj);
  assert(r < 2*obj->r0);

  *v = 0.5*obj->k*(r-obj->r0)*(r-obj->r0);
  *f = -obj->k*(r-obj->r0);

  return 0;
}
