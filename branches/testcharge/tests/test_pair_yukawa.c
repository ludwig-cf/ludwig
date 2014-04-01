/*****************************************************************************
 *
 *  test_pair_yukawa.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids_halo.h"
#include "pair_yukawa.h"

#define PAIR_EPSILON 2.0
#define PAIR_KAPPA   0.1
#define PAIR_RC      10.0

int test_pair_yukawa_suite(void);
int test_pair_yukawa1(void);
int test_pair_yukawa2(void);
int test_pair_yukawa_config1(colloids_info_t * cinfo, interact_t * interact,
			     pair_yukawa_t * pair);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  test_pair_yukawa_suite();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_yukawa_suite
 *
 *****************************************************************************/

int test_pair_yukawa_suite(void) {

  pe_init();
  coords_init();

  test_pair_yukawa1();
  test_pair_yukawa2();

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_yukawa1
 *
 *****************************************************************************/

int test_pair_yukawa1(void) {

  pair_yukawa_t * pair = NULL;
  double r = 7.0;
  double f;
  double v;

  pair_yukawa_create(&pair);
  pair_yukawa_param_set(pair, PAIR_EPSILON, PAIR_KAPPA, PAIR_RC);

  pair_yukawa_single(pair, r, &v, &f);

  assert(fabs(f - 0.019741762) < FLT_EPSILON);
  assert(fabs(v - 0.024160094) < FLT_EPSILON);

  pair_yukawa_free(pair);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_yukawa2
 *
 *****************************************************************************/

int test_pair_yukawa2(void) {

  int ncell[3] = {2, 2, 2};

  colloids_info_t * cinfo = NULL;
  interact_t * interact = NULL;
  pair_yukawa_t * pair = NULL;

  colloids_info_create(ncell, &cinfo);
  interact_create(&interact);
  pair_yukawa_create(&pair);

  assert(cinfo);
  assert(interact);
  assert(pair);

  pair_yukawa_param_set(pair, PAIR_EPSILON, PAIR_KAPPA, PAIR_RC);
  pair_yukawa_register(pair, interact);

  test_pair_yukawa_config1(cinfo, interact, pair);

  pair_yukawa_free(pair);
  interact_free(interact);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_yukawa_config1
 *
 *****************************************************************************/

int test_pair_yukawa_config1(colloids_info_t * cinfo, interact_t * interact,
			     pair_yukawa_t * pair) {

  int nc;
  double r  = 7.0;
  double f, v;
  double r1[3];
  double r2[3];
  double stats[INTERACT_STAT_MAX];
  double stats_local[INTERACT_STAT_MAX];

  colloid_t * pc1 = NULL;
  colloid_t * pc2 = NULL;

  assert(cinfo);
  assert(interact);
  assert(pair);

  r1[X] = Lmin(X) + 0.5*sqrt(1.0/3.0)*r;
  r1[Y] = Lmin(Y) + 0.5*sqrt(1.0/3.0)*r;
  r1[Z] = Lmin(Z) + 0.5*sqrt(1.0/3.0)*r;

  r2[X] = Lmin(X) + L(X) - 0.5*sqrt(1.0/3.0)*r;
  r2[Y] = Lmin(Y) + L(Y) - 0.5*sqrt(1.0/3.0)*r;
  r2[Z] = Lmin(Z) + L(Z) - 0.5*sqrt(1.0/3.0)*r;

  colloids_info_add_local(cinfo, 1, r1, &pc1);
  colloids_info_add_local(cinfo, 2, r2, &pc2);
  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);

  interact_pairwise(interact, cinfo);
  pair_yukawa_single(pair, r, &v, &f);

  f = f/sqrt(3.0); /* To get components */

  if (pe_size() == 1) {
    assert(fabs(pc1->force[X] - f) < FLT_EPSILON);
    assert(fabs(pc1->force[Y] - f) < FLT_EPSILON);
    assert(fabs(pc1->force[Z] - f) < FLT_EPSILON);
    assert(fabs(pc2->force[X] + f) < FLT_EPSILON);
    assert(fabs(pc2->force[Y] + f) < FLT_EPSILON);
    assert(fabs(pc2->force[Z] + f) < FLT_EPSILON);
  }

  pair_yukawa_stats(pair, stats_local);

  /* This is slightly over-the-top; I just really want one sum and
   * one minimum */

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_SUM,
		cart_comm());
  assert(fabs(stats[INTERACT_STAT_VLOCAL] - v) < FLT_EPSILON);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_MIN,
		cart_comm());
  assert(fabs(stats[INTERACT_STAT_RMINLOCAL] - r) < FLT_EPSILON);


  return 0;
}
