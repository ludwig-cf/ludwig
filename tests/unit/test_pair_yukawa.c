/*****************************************************************************
 *
 *  test_pair_yukawa.c
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
#include "pair_yukawa.h"
#include "tests.h"

#define PAIR_EPSILON 2.0
#define PAIR_KAPPA   0.1
#define PAIR_RC      10.0

int test_pair_yukawa1(pe_t * pe, cs_t * cs);
int test_pair_yukawa2(pe_t * pe, cs_t * cs);
int test_pair_yukawa_config1(pe_t * pe, cs_t * cs, colloids_info_t * cinfo,
			     interact_t * interact,
			     pair_yukawa_t * pair);

/*****************************************************************************
 *
 *  test_pair_yukawa_suite
 *
 *****************************************************************************/

int test_pair_yukawa_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_pair_yukawa1(pe, cs);
  test_pair_yukawa2(pe, cs);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_pair_yakawa\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_yukawa1
 *
 *****************************************************************************/

int test_pair_yukawa1(pe_t * pe, cs_t * cs) {

  pair_yukawa_t * pair = NULL;
  double r = 7.0;
  double f;
  double v;

  assert(pe);
  assert(cs);

  pair_yukawa_create(pe, cs, &pair);
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

int test_pair_yukawa2(pe_t * pe, cs_t * cs) {

  int ncell[3] = {2, 2, 2};

  colloids_info_t * cinfo = NULL;
  interact_t * interact = NULL;
  pair_yukawa_t * pair = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, ncell, &cinfo);
  interact_create(pe, cs, &interact);
  pair_yukawa_create(pe, cs, &pair);

  assert(cinfo);
  assert(interact);
  assert(pair);

  pair_yukawa_param_set(pair, PAIR_EPSILON, PAIR_KAPPA, PAIR_RC);
  pair_yukawa_register(pair, interact);

  test_pair_yukawa_config1(pe, cs, cinfo, interact, pair);

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

int test_pair_yukawa_config1(pe_t * pe, cs_t * cs, colloids_info_t * cinfo,
			     interact_t * interact,
			     pair_yukawa_t * pair) {

  int nc;
  double r  = 7.0;
  double f, v;
  double r1[3];
  double r2[3];
  double lmin[3];
  double ltot[3];
  double stats[INTERACT_STAT_MAX];
  double stats_local[INTERACT_STAT_MAX];

  colloid_t * pc1 = NULL;
  colloid_t * pc2 = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);
  assert(cinfo);
  assert(interact);
  assert(pair);

  cs_lmin(cs, lmin);
  cs_ltot(cs, ltot);
  cs_cart_comm(cs, &comm);

  r1[X] = lmin[X] + 0.5*sqrt(1.0/3.0)*r;
  r1[Y] = lmin[Y] + 0.5*sqrt(1.0/3.0)*r;
  r1[Z] = lmin[Z] + 0.5*sqrt(1.0/3.0)*r;

  r2[X] = lmin[X] + ltot[X] - 0.5*sqrt(1.0/3.0)*r;
  r2[Y] = lmin[Y] + ltot[Y] - 0.5*sqrt(1.0/3.0)*r;
  r2[Z] = lmin[Z] + ltot[Z] - 0.5*sqrt(1.0/3.0)*r;

  colloids_info_add_local(cinfo, 1, r1, &pc1);
  colloids_info_add_local(cinfo, 2, r2, &pc2);
  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);

  interact_pairwise(interact, cinfo);
  pair_yukawa_single(pair, r, &v, &f);

  f = f/sqrt(3.0); /* To get components */

  if (pe_mpi_size(pe) == 1) {
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
		comm);
  assert(fabs(stats[INTERACT_STAT_VLOCAL] - v) < FLT_EPSILON);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_MIN,
		comm);
  assert(fabs(stats[INTERACT_STAT_RMINLOCAL] - r) < FLT_EPSILON);


  return 0;
}
