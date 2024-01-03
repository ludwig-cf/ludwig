/*****************************************************************************
 *
 *  test_interaction.c
 *
 *  Actually for some of the single-particle routines.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "interaction.h"

int test_colloids_update_forces_external(pe_t * pe);
int test_colloids_update_forces_fluid_body_force(pe_t * pe);

/*****************************************************************************
 *
 *  test_interaction_suite
 *
 *****************************************************************************/

int test_interaction_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_colloids_update_forces_external(pe);
  test_colloids_update_forces_fluid_body_force(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloids_update_forces_external
 *
 *****************************************************************************/

int test_colloids_update_forces_external(pe_t * pe) {

  int ifail = 0;
  int ncells[3] = {8, 8, 8};
  cs_t * cs = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  double s[3] = {1.0, 0.0, 0.0};  /* A magnetic dipole */

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, ncells, &cinfo);

  {
    /* Add a sample colloid to list */
    int index = 1;
    double r0[3] = {2.0, 2.0, 2.0};

    colloids_info_add_local(cinfo, index, r0, &pc);
    if (pc) {
      pc->s.s[X] = s[X]; pc->s.s[Y] = s[Y]; pc->s.s[Z] = s[Z];
    }
    colloids_info_update_lists(cinfo);
  }

  {
    /* Gravitation/sedimentation */

    physics_t * phys = NULL;
    double fg[3] = {0.01, 0.02, 0.03};
    physics_create(pe, &phys);
    physics_fgrav_set(phys, fg);

    colloids_update_forces_external(cinfo, phys);
    if (pc) {
      assert(fabs(pc->force[X] - fg[X]) < DBL_EPSILON);
      assert(fabs(pc->force[Y] - fg[Y]) < DBL_EPSILON);
      assert(fabs(pc->force[Z] - fg[Z]) < DBL_EPSILON);
    }
    physics_free(phys);
  }

  {
    /* Magnetic torque */

    physics_t * phys = NULL;
    double b0[3] = {0.01, 0.02, 0.03};
    physics_create(pe, &phys);
    physics_b0_set(phys, b0);
    colloids_update_forces_external(cinfo, phys);
    if (pc) {
      assert(fabs(pc->torque[X] - (s[Y]*b0[Z] - s[Z]*b0[Y])) < DBL_EPSILON);
      assert(fabs(pc->torque[Y] - (s[Z]*b0[X] - s[X]*b0[Z])) < DBL_EPSILON);
      assert(fabs(pc->torque[Z] - (s[X]*b0[Y] - s[Y]*b0[X])) < DBL_EPSILON);
    }
    physics_free(phys);
  }

  colloids_info_free(cinfo);
  cs_free(cs);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloids_update_forces_fluid_body_force
 *
 *****************************************************************************/

int test_colloids_update_forces_fluid_body_force(pe_t * pe) {

  int ifail = 0;
  int ncells[3] = {8, 8, 8};
  cs_t * cs = NULL;
  colloid_t * pc = NULL;
  colloids_info_t * cinfo = NULL;

  cs_create(pe, &cs);
  cs_init(cs);
  colloids_info_create(pe, cs, ncells, &cinfo);

  {
    /* Add a sample colloid to list */
    int index = 1;
    double r0[3] = {2.0, 2.0, 2.0};
    double a0    = 0.5; /* discrete volume should be unity */
    colloids_info_add_local(cinfo, index, r0, &pc);
    if (pc) {
      pc->s.a0 = a0;
      pc->s.ah = a0;
    }
    colloids_info_update_lists(cinfo);
  }

  {
    /* Gravity check: no body force contribution on colloid... */

    physics_t * phys = NULL;
    double fb[3] = {0.01, 0.02, 0.03};
    double fg[3] = {0.04, 0.00, 0.00};
    physics_create(pe, &phys);
    physics_fbody_set(phys, fb);
    physics_fgrav_set(phys, fg);

    colloids_update_forces_fluid_body_force(cinfo, phys);
    if (pc) {
      assert(fabs(pc->force[X] - 0.0) < DBL_EPSILON);
      assert(fabs(pc->force[Y] - 0.0) < DBL_EPSILON);
      assert(fabs(pc->force[Z] - 0.0) < DBL_EPSILON);
    }
    physics_free(phys);
  }

  {
    /* No gravity, body force... */
    physics_t * phys = NULL;
    double fb[3] = {0.01, 0.02, 0.03};
    physics_create(pe, &phys);
    physics_fbody_set(phys, fb);

    colloids_update_forces_fluid_body_force(cinfo, phys);
    if (pc) {
      assert(fabs(pc->force[X] - fb[X]) < DBL_EPSILON);
      assert(fabs(pc->force[Y] - fb[Y]) < DBL_EPSILON);
      assert(fabs(pc->force[Z] - fb[Z]) < DBL_EPSILON);
    }

    physics_free(phys);
  }

  colloids_info_free(cinfo);
  cs_free(cs);

  return ifail;
}
