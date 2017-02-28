/*****************************************************************************
 *
 *  test_angle_cosine.c
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
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids_halo.h"
#include "angle_cosine.h"
#include "tests.h"

#define ANGLE_KAPPA 2.0

int test_angle_cosine1(pe_t * pe, cs_t * cs);
int test_angle_cosine2(pe_t * pe, cs_t * cs);
int test_create_trimer(colloids_info_t * cinfo, double a, double r1[3],
		       double r2[3], double r3[3], colloid_t * pc[3]);

/*****************************************************************************
 *
 *  test_angle_cosine_suite
 *
 *****************************************************************************/

int test_angle_cosine_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_angle_cosine1(pe, cs);
  test_angle_cosine2(pe, cs);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_angle_cosine\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_angle_cosine1
 *
 *  This trimer has zero angle, so no bending force.
 *
 *****************************************************************************/

int test_angle_cosine1(pe_t * pe, cs_t * cs) {

  int ncell[3] = {2, 2, 2};
  double a = 2.3;                   /* colloid size */
  double r1[3] = {1.0, 1.0, 1.0};   /* position 1 */
  double r2[3] = {1.0, 1.0, 2.0};   /* position 2 */
  double r3[3] = {1.0, 1.0, 3.0};   /* position 3 */
  colloid_t * pc3[3];

  colloids_info_t * cinfo = NULL;
  interact_t * interact = NULL;
  angle_cosine_t * angle = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, ncell, &cinfo);
  interact_create(pe, cs, &interact);
  angle_cosine_create(pe, cs, &angle);
  angle_cosine_param_set(angle, ANGLE_KAPPA);
  angle_cosine_register(angle, interact);

  test_create_trimer(cinfo, a, r1, r2, r3, pc3);
  interact_find_bonds(interact, cinfo);
  interact_angles(interact, cinfo);

  if (pe_mpi_size(pe) == 1) {
    assert(fabs(pc3[0]->force[X] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[0]->force[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[0]->force[Z] - 0.0) < DBL_EPSILON);

    assert(fabs(pc3[1]->force[X] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[1]->force[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[1]->force[Z] - 0.0) < DBL_EPSILON);

    assert(fabs(pc3[2]->force[X] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[2]->force[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(pc3[2]->force[Z] - 0.0) < DBL_EPSILON);
  }

  angle_cosine_free(angle);
  interact_free(interact);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_angle_cosine2
 *
 *  This trimer has angle 45 degrees, with force (1/sqrt(2))kappa
 *
 *****************************************************************************/

int test_angle_cosine2(pe_t * pe, cs_t * cs) {

  int ncell[3] = {2, 2, 2};
  double a = 2.3;                   /* colloid size */
  double r1[3] = {1.0, 1.0, 1.0};   /* position 1 */
  double r2[3] = {1.0, 2.0, 1.0};   /* position 2 */
  double r3[3] = {2.0, 1.0, 1.0};   /* position 3 */
  double fexpect;

  colloid_t * pc3[3];

  colloids_info_t * cinfo = NULL;
  interact_t * interact = NULL;
  angle_cosine_t * angle = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, ncell, &cinfo);
  interact_create(pe, cs, &interact);
  angle_cosine_create(pe, cs, &angle);
  angle_cosine_param_set(angle, ANGLE_KAPPA);
  angle_cosine_register(angle, interact);

  test_create_trimer(cinfo, a, r1, r2, r3, pc3);
  interact_find_bonds(interact, cinfo);
  interact_angles(interact, cinfo);

  /* A comunication would be required in parallel */

  if (pe_mpi_size(pe) == 1) {
    fexpect = sqrt(2.0);

    assert(fabs(pc3[0]->force[X] - -fexpect) < FLT_EPSILON);
    assert(fabs(pc3[0]->force[Y] - 0.0) < FLT_EPSILON);
    assert(fabs(pc3[0]->force[Z] - 0.0) < FLT_EPSILON);


    fexpect = 1.0/sqrt(2.0);

    assert(fabs(pc3[1]->force[X] -  fexpect) < FLT_EPSILON);
    assert(fabs(pc3[1]->force[Y] - -fexpect) < FLT_EPSILON);
    assert(fabs(pc3[1]->force[Z] - 0.0) < FLT_EPSILON);

    assert(fabs(pc3[2]->force[X] - fexpect) < FLT_EPSILON);
    assert(fabs(pc3[2]->force[Y] - fexpect) < FLT_EPSILON);
    assert(fabs(pc3[2]->force[Z] - 0.0) < FLT_EPSILON);
  }

  angle_cosine_free(angle);
  interact_free(interact);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_create_trimer
 *
 *****************************************************************************/

int test_create_trimer(colloids_info_t * cinfo, double a, double r1[3],
		       double r2[3], double r3[3], colloid_t * pc[3]) {

  int nc = 0;

  assert(cinfo);
  assert(pc);

  colloids_info_add_local(cinfo, 1, r1, pc);
  if (pc[0]) {
    pc[0]->s.a0 = a;
    pc[0]->s.ah = a;
    pc[0]->s.nbonds = 1;
    pc[0]->s.bond[0] = 2;
  }

  colloids_info_add_local(cinfo, 2, r2, pc + 1);
  if (pc[1]) {
    pc[1]->s.a0 = a;
    pc[1]->s.ah = a;
    pc[1]->s.nbonds = 2;
    pc[1]->s.bond[0] = 1;
    pc[1]->s.bond[1] = 3;
    pc[1]->s.nangles = 1;
  }

  colloids_info_add_local(cinfo, 3, r3, pc + 2);
  if (pc[2]) {
    pc[2]->s.a0 = a;
    pc[2]->s.ah = a;
    pc[2]->s.nbonds = 1;
    pc[2]->s.bond[0] = 2;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 3);

  colloids_halo_state(cinfo);
  colloids_info_list_local_build(cinfo);

  return 0;
}
