/*****************************************************************************
 *
 *  test_pair_ss_cut.c
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
#include "pair_ss_cut.h"

#define PAIR_EPSILON 0.001
#define PAIR_SIGMA   0.8
#define PAIR_NU      2
#define PAIR_HC      0.25

int test_pair_ss_cut_suite(void);
int test_pair_ss_cut1(void);
int test_pair_ss_cut2(void);
int test_pair_config1(colloids_info_t * cinfo, interact_t * interact,
		      pair_ss_cut_t * pair);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  test_pair_ss_cut_suite();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_ss_cut_suite
 *
 *****************************************************************************/

int test_pair_ss_cut_suite(void) {

  pe_init();
  coords_init();

  test_pair_ss_cut1();
  test_pair_ss_cut2();

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_ss_cut1
 *
 *****************************************************************************/

int test_pair_ss_cut1(void) {

  pair_ss_cut_t * pair = NULL;
  double h, f, v;

  pair_ss_cut_create(&pair);
  assert(pair);

  pair_ss_cut_param_set(pair, PAIR_EPSILON, PAIR_SIGMA, PAIR_NU, PAIR_HC);

  h = 0.0125;
  pair_ss_cut_single(pair, h, &f, &v);
  assert(fabs(f - 655.27808) < FLT_EPSILON);
  assert(fabs(v - 4.0663040) < FLT_EPSILON);

  h = PAIR_HC;
  pair_ss_cut_single(pair, h, &f, &v);
  assert(fabs(f - 0.0) < DBL_EPSILON);
  assert(fabs(v - 0.0) < DBL_EPSILON);

  pair_ss_cut_free(pair);

  return 0;
}

/*****************************************************************************
 *
 *  test_ss_cut2
 *
 *****************************************************************************/

int test_pair_ss_cut2(void) {

  int ncell[3] = {2, 2, 2};

  colloids_info_t * cinfo = NULL;
  interact_t * interact = NULL;
  pair_ss_cut_t * pair = NULL;

  colloids_info_create(ncell, &cinfo);
  assert(cinfo);
  interact_create(&interact);
  assert(interact);

  pair_ss_cut_create(&pair);
  assert(pair);

  pair_ss_cut_param_set(pair, PAIR_EPSILON, PAIR_SIGMA, PAIR_NU, PAIR_HC);
  pair_ss_cut_register(pair, interact);

  test_pair_config1(cinfo, interact, pair);

  /* Finish */

  pair_ss_cut_free(pair);
  interact_free(interact);
  colloids_info_free(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_config1
 *
 *****************************************************************************/

int test_pair_config1(colloids_info_t * cinfo, interact_t * interact,
		      pair_ss_cut_t * pair) {

  int nc;
  double a0 = 2.3;
  double ah = 2.3;
  double dh = 0.1;
  double h, f, v;
  double r1[3];
  double r2[3];
  double stats[INTERACT_STAT_MAX];
  double stats_local[INTERACT_STAT_MAX];

  colloid_t * pc1 = NULL;
  colloid_t * pc2 = NULL;

  assert(cinfo);
  assert(interact);
  assert(pair);

  h = 2*ah + dh;
  /* The sqrt(3)*sqrt(4) gives separation h in 3d */
  r1[X] = Lmin(X) + h/sqrt(12.0);
  r1[Y] = Lmin(Y) + h/sqrt(12.0);
  r1[Z] = Lmin(Z) + h/sqrt(12.0);

  colloids_info_add_local(cinfo, 1, r1, &pc1);
  if (pc1) {
    pc1->s.a0 = a0;
    pc1->s.ah = ah;
  }

  r2[X] = Lmin(X) + L(X) - h/sqrt(12.0);
  r2[Y] = Lmin(Y) + L(Y) - h/sqrt(12.0);
  r2[Z] = Lmin(Z) + L(Z) - h/sqrt(12.0);

  colloids_info_add_local(cinfo, 2, r2, &pc2);
  if (pc2) {
    pc2->s.a0 = a0;
    pc2->s.ah = ah;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);

  /* Compute interactions and compare against single version */

  interact_pairwise(interact, cinfo);
  pair_ss_cut_single(pair, dh, &f, &v);

  f = f/sqrt(3.0);
 
  if (pe_size() == 1) {
    assert(fabs(pc1->force[X] - f) < FLT_EPSILON);
    assert(fabs(pc1->force[Y] - f) < FLT_EPSILON);
    assert(fabs(pc1->force[Z] - f) < FLT_EPSILON);

    assert(fabs(pc2->force[X] + f) < FLT_EPSILON);
    assert(fabs(pc2->force[Y] + f) < FLT_EPSILON);
    assert(fabs(pc2->force[Z] + f) < FLT_EPSILON);
  }

  pair_ss_cut_stats(pair, stats_local);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_SUM,
		cart_comm());
  assert(fabs(stats[INTERACT_STAT_VLOCAL] - v) < FLT_EPSILON);

  return 0;
}
