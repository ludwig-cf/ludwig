/*****************************************************************************
 *
 *  test_lubrication.c
 *
 *  Sphere-sphere lubrication corrections.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "ran.h"
#include "coords.h"
#include "lubrication.h"

int test_lubrication_suite(void);
int test_lubrication_ss_fnorm(void);
int test_lubrication_ss_ftang(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  test_lubrication_suite();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_lubrication_suite
 *
 *****************************************************************************/

int test_lubrication_suite(void) {

  physics_t * physics = NULL;

  pe_init();
  coords_init();
  physics_ref(&physics);
  ran_init();

  test_lubrication_ss_fnorm();
  test_lubrication_ss_ftang();

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_lubrication_ss_fnorm
 *
 *  Force normal component (no kt)
 *
 *****************************************************************************/

int test_lubrication_ss_fnorm(void) {

  double rc = 0.75;
  double rchmax;
  lubr_t * lubr = NULL;

  /* Test data */

  double a1 = 1.25;
  double a2 = 2.3;
  double u1[3] = {0.0, 0.0, 0.0};
  double u2[3] = {0.0, 0.0, 0.0};
  double r12[3] = {0.0, 0.0, 4.0};
  double factual[3];
  double fexpect[3];

  lubrication_create(&lubr);
  assert(lubr);

  lubrication_rch_set(lubr, LUBRICATION_SS_FNORM, rc);
  lubrication_rchmax(lubr, &rchmax);
  assert(fabs(rc - rchmax) < DBL_EPSILON);

  /* Both zero velocity */

  fexpect[X] = 0.0; fexpect[Y] = 0.0; fexpect[Z] = 0.0;

  lubrication_single(lubr, a1, a2, u1, u2, r12, factual);
  assert(fabs(factual[X] - fexpect[X]) < FLT_EPSILON);
  assert(fabs(factual[Y] - fexpect[Y]) < FLT_EPSILON);
  assert(fabs(factual[Z] - fexpect[Z]) < FLT_EPSILON);

  /* Finite velocity test values */

  u1[Z] = 0.5;
  fexpect[Z] = -1.0035643;

  lubrication_single(lubr, a1, a2, u1, u2, r12, factual);
  assert(fabs(factual[Z] - fexpect[Z]) < FLT_EPSILON);

  lubrication_free(lubr);

  return 0;
}

/*****************************************************************************
 *
 *  test_lubrication_ss_ftang
 *
 *  Tangential component of force
 *
 *****************************************************************************/

int test_lubrication_ss_ftang(void) {

  double rc = 0.25;
  double rchmax;
  lubr_t * lubr = NULL;

  /* Test data */

  double a1 = 1.25;
  double a2 = 2.3;
  double u1[3] = {0.0, 0.0, 0.0};
  double u2[3] = {0.0, 0.0, 0.0};
  double r12[3] = {0.0, 0.0, 3.65};
  double fexpect[3] = {0.0, 0.0, 0.0};
  double factual[3];

  lubrication_create(&lubr);
  assert(lubr);

  lubrication_rch_set(lubr, LUBRICATION_SS_FTANG, rc);
  lubrication_rchmax(lubr, &rchmax);
  assert(fabs(rc - rchmax) < DBL_EPSILON);

  /* Both zero velocity */

  lubrication_single(lubr, a1, a2, u1, u2, r12, factual);
  assert(fabs(fexpect[X] - factual[X]) < FLT_EPSILON);
  assert(fabs(fexpect[Y] - factual[Y]) < FLT_EPSILON);
  assert(fabs(fexpect[Z] - factual[Z]) < FLT_EPSILON);

  /* Finite tangential velocity test values */

  u1[X] = 0.5;
  fexpect[X] = -0.40893965;

  lubrication_single(lubr, a1, a2, u1, u2, r12, factual);
  assert(fabs(fexpect[X] - factual[X]) < FLT_EPSILON);

  lubrication_free(lubr);

  return 0;
}
