/*****************************************************************************
 *
 *  test_pair_lj_cut.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2025 The University of Edinburgh
 *
 *  Contributing authors;
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloid.h"
#include "colloids.h"
#include "colloids_halo.h"
#include "pair_lj_cut.h"

#define PAIR_EPSILON 1.0
#define PAIR_SIGMA   1.0
#define PAIR_RC      3.0

int test_pair_lj_cut1(pe_t * pe, cs_t * cs);
int test_pair_lj_cut2(pe_t * pe, cs_t * cs);
int test_pair_lj_cut2_with_state(pe_t * pe, cs_t * cs);
int test_pair_config1(colloids_info_t * cinfo, interact_t * interact,
		      pair_lj_cut_t * lj);
int test_pair_config1_with_state(colloids_info_t * cinfo, interact_t * interact,
		      pair_lj_cut_t * lj);

void print_colloid_data(colloid_t *pc1);

/*****************************************************************************
 *
 *  test_pair_lj_cut_suite
 *
 *****************************************************************************/

int test_pair_lj_cut_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_pair_lj_cut1(pe, cs);
  test_pair_lj_cut2(pe, cs);
  test_pair_lj_cut2_with_state(pe, cs);

  cs_free(cs);
  pe_info(pe, "PASS     ./unit/test_pair_lj_cut\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_lj_cut1
 *
 *****************************************************************************/

int test_pair_lj_cut1(pe_t * pe, cs_t * cs) {

  pair_lj_cut_t * lj = NULL;
  double h, f, v;

  pair_lj_cut_create(pe, cs, &lj);
  assert(lj);

  pair_lj_cut_param_set(lj, PAIR_EPSILON, PAIR_SIGMA, PAIR_RC);

  h = PAIR_SIGMA;
  pair_lj_cut_single(lj, h, &f, &v);
  assert(fabs(f - 24.01094383) < FLT_EPSILON);
  assert(fabs(v -  0.027367102) < FLT_EPSILON);

  h = pow(2.0, 1.0/6.0)*PAIR_SIGMA;
  pair_lj_cut_single(lj, h, &f, &v);
  assert(fabs(f - 0.010943830) < FLT_EPSILON);
  assert(fabs(v - -0.97397310) < FLT_EPSILON);

  h = PAIR_RC;
  pair_lj_cut_single(lj, h, &f, &v);
  assert(fabs(f - 0.0) < FLT_EPSILON);
  assert(fabs(v - 0.0) < FLT_EPSILON);

  pair_lj_cut_free(lj);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_lj_cut2
 *
 *****************************************************************************/

int test_pair_lj_cut2(pe_t * pe, cs_t * cs) {


  colloid_options_t opts  = colloid_options_default();
  colloids_info_t * cinfo = NULL;

  interact_t * interact = NULL;
  pair_lj_cut_t * lj = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, &opts, &cinfo);
  interact_create(pe, cs, &interact);
  pair_lj_cut_create(pe, cs, &lj);

  assert(cinfo);
  assert(interact);
  assert(lj);

  pair_lj_cut_param_set(lj, PAIR_EPSILON, PAIR_SIGMA, PAIR_RC);
  pair_lj_cut_register(lj, interact);

  test_pair_config1(cinfo, interact, lj);

  /* Finish */

  pair_lj_cut_free(lj);
  interact_free(interact);
  colloids_info_free(&cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_lj_cut2_with_state
 *
 *****************************************************************************/

int test_pair_lj_cut2_with_state(pe_t * pe, cs_t * cs) {


  colloid_options_t opts  = colloid_options_default();
  colloids_info_t * cinfo = NULL;

  interact_t * interact = NULL;
  pair_lj_cut_t * lj = NULL;

  assert(pe);
  assert(cs);

  colloids_info_create(pe, cs, &opts, &cinfo);
  interact_create(pe, cs, &interact);
  pair_lj_cut_create(pe, cs, &lj);

  assert(cinfo);
  assert(interact);
  assert(lj);

  pair_lj_cut_param_set(lj, PAIR_EPSILON, PAIR_SIGMA, PAIR_RC);
  pair_lj_cut_register(lj, interact);

  test_pair_config1_with_state(cinfo, interact, lj);

  /* Finish */

  pair_lj_cut_free(lj);
  interact_free(interact);
  colloids_info_free(&cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_config1
 *
 *****************************************************************************/

int test_pair_config1(colloids_info_t * cinfo,
		      interact_t * interact,
		      pair_lj_cut_t * lj) {

  int nc;
  double a0 = 1.25;
  double ah = 1.25;
  double dh = 0.1;
  double h, f, v;
  double r1[3];
  double r2[3];
  double ltot[3];
  double stats[INTERACT_STAT_MAX];
  double stats_local[INTERACT_STAT_MAX];

  MPI_Comm comm;

  colloid_t * pc1 = NULL;
  colloid_t * pc2 = NULL;

  assert(cinfo);
  assert(interact);
  assert(lj);

  cs_ltot(cinfo->cs, ltot);
  cs_cart_comm(cinfo->cs, &comm);

  h = 2.0*ah + dh;
  r1[X] = 0.5*ltot[X] - 0.5*h;
  r1[Y] = 0.5*ltot[Y];
  r1[Z] = 0.5*ltot[Z];

  colloids_info_add_local(cinfo, 1, r1, a0, &pc1);
  //print_colloid_data(pc1);
  if (pc1) {
    pc1->s.a0 = a0;
    pc1->s.ah = ah;
  }

  r2[X] = r1[X] + h;
  r2[Y] = r1[Y];
  r2[Z] = r1[Z];

  colloids_info_add_local(cinfo, 2, r2, a0, &pc2);
  //print_colloid_data(pc2);
  if (pc2) {
    pc2->s.a0 = a0;
    pc2->s.ah = ah;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);

  /* Check interaction against single version */

  interact_pairwise(interact, cinfo);
  pair_lj_cut_single(lj, h, &f, &v);

  if (pe_mpi_size(cinfo->pe) == 1) {
    assert(fabs(pc1->force[X] - 0.018743896) < FLT_EPSILON);
    assert(fabs(pc1->force[Y] - 0.0)         < FLT_EPSILON);
    assert(fabs(pc1->force[Z] - 0.0)         < FLT_EPSILON);

    assert(fabs(pc2->force[X] + 0.018743896) < FLT_EPSILON);
    assert(fabs(pc2->force[Y] + 0.0)         < FLT_EPSILON);
    assert(fabs(pc2->force[Z] + 0.0)         < FLT_EPSILON);
  }

  pair_lj_cut_stats(lj, stats_local);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_SUM,
		comm);

  //printf("stats - v %f %f diff %f epsilon %f\n", stats[INTERACT_STAT_VLOCAL], v, fabs(stats[INTERACT_STAT_VLOCAL] - v), FLT_EPSILON);
  assert(fabs(stats[INTERACT_STAT_VLOCAL] - v) < FLT_EPSILON);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_MIN,
		comm);

  assert(fabs(stats[INTERACT_STAT_RMINLOCAL] - h) < FLT_EPSILON);
  assert(fabs(stats[INTERACT_STAT_HMINLOCAL] - dh) < FLT_EPSILON);

  return 0;
}

/*****************************************************************************
 *
 *  test_pair_config1_with_state
 *
 *****************************************************************************/

int test_pair_config1_with_state(colloids_info_t * cinfo,
		      interact_t * interact,
		      pair_lj_cut_t * lj) {

  int nc;
  double a0 = 1.25;
  double ah = 1.25;
  double dh = 0.1;
  double h, f, v;
  double r1[3];
  double r2[3];
  double ltot[3];
  double stats[INTERACT_STAT_MAX];
  double stats_local[INTERACT_STAT_MAX];
  colloid_state_t state1, state2;

  MPI_Comm comm;

  colloid_t * pc1 = NULL;
  colloid_t * pc2 = NULL;
  colloid_t * pc3 = NULL;
  colloid_t * pc4 = NULL;

  assert(cinfo);
  assert(interact);
  assert(lj);

  cs_ltot(cinfo->cs, ltot);
  cs_cart_comm(cinfo->cs, &comm);

  h = 2.0*ah + dh;
  r1[X] = 0.5*ltot[X] - 0.5*h;
  r1[Y] = 0.5*ltot[Y];
  r1[Z] = 0.5*ltot[Z];

  create_dummy_state(&state1, 1, a0, r1);

  colloids_info_add_local_with_state(cinfo, &state1, &pc1);
  //print_colloid_data(pc1);
  if (pc1) {
    pc1->s.a0 = a0;
    pc1->s.ah = ah;
  }

  r2[X] = r1[X] + h;
  r2[Y] = r1[Y];
  r2[Z] = r1[Z];

  create_dummy_state(&state2, 2, a0, r2);

  colloids_info_add_local_with_state(cinfo, &state2, &pc2);
  //print_colloid_data(pc2);
  if (pc2) {
    pc2->s.a0 = a0;
    pc2->s.ah = ah;
  }

  colloids_info_ntotal_set(cinfo);
  colloids_info_ntotal(cinfo, &nc);
  assert(nc == 2);

  colloids_halo_state(cinfo);

  /* Check interaction against single version */

  interact_pairwise(interact, cinfo);
  pair_lj_cut_single(lj, h, &f, &v);

  if (pe_mpi_size(cinfo->pe) == 1) {
    assert(fabs(pc1->force[X] - 0.018743896) < FLT_EPSILON);
    assert(fabs(pc1->force[Y] - 0.0)         < FLT_EPSILON);
    assert(fabs(pc1->force[Z] - 0.0)         < FLT_EPSILON);

    assert(fabs(pc2->force[X] + 0.018743896) < FLT_EPSILON);
    assert(fabs(pc2->force[Y] + 0.0)         < FLT_EPSILON);
    assert(fabs(pc2->force[Z] + 0.0)         < FLT_EPSILON);
  }

  pair_lj_cut_stats(lj, stats_local);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_SUM,
		comm);

  //printf("stats - v %f %f diff %f epsilon %f\n", stats[INTERACT_STAT_VLOCAL], v, fabs(stats[INTERACT_STAT_VLOCAL] - v), FLT_EPSILON);
  assert(fabs(stats[INTERACT_STAT_VLOCAL] - v) < FLT_EPSILON);

  MPI_Allreduce(stats_local, stats, INTERACT_STAT_MAX, MPI_DOUBLE, MPI_MIN,
		comm);

  assert(fabs(stats[INTERACT_STAT_RMINLOCAL] - h) < FLT_EPSILON);
  assert(fabs(stats[INTERACT_STAT_HMINLOCAL] - dh) < FLT_EPSILON);

  return 0;
}

// Print the colloid state and other properties.
void print_colloid_data(colloid_t *pc1) {

  //assert(pc1);

  printf("Colloid index: %d\n", pc1->s.index);
  printf("Colloid a0: %f\n", pc1->s.a0);
  printf("Colloid ah: %f\n", pc1->s.ah);
  printf("Colloid position: (%f, %f, %f)\n", pc1->s.r[0], pc1->s.r[1], pc1->s.r[2]);
  //printf("Colloid force: (%f, %f, %f)\n", pc1->force[0], pc1->force[1], pc1->force[2]);
  //printf("Colloid torque: (%f, %f, %f)\n", pc1->torque[0], pc1->torque[1], pc1->torque[2]);
  //printf("Colloid f0: (%f, %f, %f)\n", pc1->f0[0], pc1->f0[1], pc1->f0[2]);
  //printf("Colloid t0: (%f, %f, %f)\n", pc1->t0[0], pc1->t0[1], pc1->t0[2]);
  //printf("Colloid cbar: (%f, %f, %f)\n", pc1->cbar[0], pc1->cbar[1], pc1->cbar[2]);
  //printf("Colloid rxcbar: (%f, %f, %f)\n", pc1->rxcbar[0], pc1->rxcbar[1], pc1->rxcbar[2]);
  //printf("Colloid deltam: %f\n", pc1->deltam);
  //printf("Colloid sumw: %f\n", pc1->sumw);
  //printf("Colloid sump: %f\n", pc1->sump);
  //printf("Colloid dq: (%f, %f)\n", pc1->dq[0], pc1->dq[1]);
  //printf("Colloid zeta: \n");
  //for (int i = 0; i < 21; i++) {
  //  printf("%d %f\n", i, pc1->zeta[i]);
  //}
  //printf("\n");

  return;
}