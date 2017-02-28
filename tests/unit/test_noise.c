/*****************************************************************************
 *
 *  test_noise.c
 *
 *  Test the basic lattice noise generator type.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2016 The University of Edinburgh
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
#include "noise.h"
#include "tests.h"

static int do_test_noise1(pe_t * pe);
static int do_test_noise2(pe_t * pe);
static int do_test_noise3(pe_t * pe);

/*****************************************************************************
 *
 *  test_noise_suite
 * 
 *****************************************************************************/

int test_noise_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* info("Noise tests\n\n");*/

  do_test_noise1(pe);
  do_test_noise2(pe);
  do_test_noise3(pe);

  pe_info(pe, "PASS     ./unit/test_noise\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_noise1
 *
 *  This is a unit test which checks the interface is working and
 *  the discrete values are correct.
 *
 *****************************************************************************/

static int do_test_noise1(pe_t * pe) {

  noise_t * noise = NULL;
  cs_t * cs = NULL;

  double a1, a2;
  double r[NNOISE_MAX];
  unsigned int state_ref[NNOISE_STATE] = {123, 456, 78, 9};
  unsigned int state[NNOISE_STATE] = {0, 0, 0, 0};

  assert(pe);
  assert(NNOISE_MAX == 10);
  assert(NNOISE_STATE == 4);

  a1 = sqrt(2.0 + sqrt(2.0));
  a2 = sqrt(2.0 - sqrt(2.0));

  cs_create(pe, &cs);
  cs_init(cs);

  noise_create(pe, cs, &noise);
  assert(noise);
  noise_init(noise, 0);

  /* The initial state[] is zero, one iteration should
   * move us to... */ 

  noise_uniform(state);

  assert(state[0] == 1234567);
  assert(state[1] == 0);
  assert(state[2] == 0);
  assert(state[3] == 0);

  /* Set some state and make sure reap vector is correct */

  noise_state_set(noise, 0, state_ref);
  noise_state(noise, 0, state);

  assert(state[0] == state_ref[0]);
  assert(state[1] == state_ref[1]);
  assert(state[2] == state_ref[2]);
  assert(state[3] == state_ref[3]);

  noise_reap(noise, 0, r);

  assert(fabs(r[0] - +a1) < DBL_EPSILON);
  assert(fabs(r[1] - -a1) < DBL_EPSILON);
  assert(fabs(r[2] - 0.0) < DBL_EPSILON);
  assert(fabs(r[3] - +a2) < DBL_EPSILON);
  assert(fabs(r[4] - +a2) < DBL_EPSILON);
  assert(fabs(r[5] - 0.0) < DBL_EPSILON);
  assert(fabs(r[6] - 0.0) < DBL_EPSILON);
  assert(fabs(r[7] - 0.0) < DBL_EPSILON);
  assert(fabs(r[8] - 0.0) < DBL_EPSILON);
  assert(fabs(r[9] - 0.0) < DBL_EPSILON);

  noise_free(noise);
  cs_free(cs);

  return 0;
}


/*****************************************************************************
 *
 *  do_test_noise2
 *
 *  Test the parallel initialisation. Check some statistics in the
 *  spatial average.
 *
 *****************************************************************************/

static int do_test_noise2(pe_t * pe) {

  int nlocal[3];
  int ic, jc, kc, index;
  int ir;

  cs_t * cs = NULL;
  noise_t * noise = NULL;

  double ltot[3];
  double r[NNOISE_MAX];
  double rstat[2], rstat_local[2] = {0.0, 0.0};
  MPI_Comm comm;

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  cs_cart_comm(cs, &comm);

  noise_create(pe, cs, &noise);
  noise_init(noise, 0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	noise_reap(noise, index, r);

	for (ir = 0; ir < NNOISE_MAX; ir++) {
	  rstat_local[0] += r[ir];
	  rstat_local[1] += r[ir]*r[ir];
	}
      }
    }
  }

  /* Mean and variance */

  MPI_Allreduce(rstat_local, rstat, 2, MPI_DOUBLE, MPI_SUM, comm);

  rstat[0] = rstat[0]/(ltot[X]*ltot[Y]*ltot[Z]);
  rstat[1] = rstat[1]/(NNOISE_MAX*ltot[X]*ltot[Y]*ltot[Z]) - rstat[0]*rstat[0];

  /* These are the results for the default seeds, system size */
  assert(fabs(rstat[0] - 4.10105573e-03) < FLT_EPSILON);
  assert(fabs(rstat[1] - 1.00177840)     < FLT_EPSILON);

  noise_free(noise);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_noise3
 *
 *  This checks up to 6th moment, which should see the following:
 *
 *  0th   \sum_i is the number of time steps times NNOISE_MAX
 *  1st   \sum_i x_i^1   = 0.0 (mean)
 *  2nd   \sum_i x_i^2   = 1.0 (variance)
 *  3rd   \sum_i x_i^3   = 0.0
 *  4th   \sum_i x_i^4   = 3.0
 *  5th   \sum_i x_i^5   = 0.0
 *  6th   \sum_i x_i^6   = 10.0
 *
 *  The statistics are computed at each lattice site. We can only do
 *  a small system (4x4x4) for as many as 1 million steps. All the
 *  moments are correct to a modest tolerance as this point.
 *
 *****************************************************************************/

static int do_test_noise3(pe_t * pe) {

  int ic, jc, kc, index;
  int ntotal[3] = {4, 4, 4};
  int nlocal[3];
  int n, nt, nsites;

  double * moment6;
  double   rnorm;
  double m1, m2, m3, m4, m5, m6;

  double r[NNOISE_MAX];
  cs_t * cs = NULL;
  noise_t * noise = NULL;

  assert(pe);

  /* Extent of the test */
  const int ntimes = 1000000;
  const double tolerance = 0.05;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);
  cs_nlocal(cs, nlocal);
  cs_nsites(cs, &nsites);

  noise_create(pe, cs, &noise);
  noise_init(noise, 0);

  moment6 = (double *) calloc(6*nsites, sizeof(double));
  assert(moment6);

  /* Loop */

  for (nt = 0; nt < ntimes; nt++) {

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = cs_index(cs, ic, jc, kc);
	  noise_reap(noise, index, r);

	  for (n = 0; n < NNOISE_MAX; n++) {
	    moment6[0*nsites + index] += r[n];
	    moment6[1*nsites + index] += r[n]*r[n];
	    moment6[2*nsites + index] += r[n]*r[n]*r[n];
	    moment6[3*nsites + index] += r[n]*r[n]*r[n]*r[n];
	    moment6[4*nsites + index] += r[n]*r[n]*r[n]*r[n]*r[n];
	    moment6[5*nsites + index] += r[n]*r[n]*r[n]*r[n]*r[n]*r[n];
	  }

	  /* Next site. */
	}
      }
    }
    /* Next time step */
  }

  /* Stats. */

  rnorm = 1.0/((double) ntimes*NNOISE_MAX);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	/* Moments */
	m1 = rnorm*moment6[0*nsites + index];
	m2 = rnorm*moment6[1*nsites + index];
	m3 = rnorm*moment6[2*nsites + index];
	m4 = rnorm*moment6[3*nsites + index];
	m5 = rnorm*moment6[4*nsites + index];
	m6 = rnorm*moment6[5*nsites + index];

	assert(fabs(m1 - 0.0)  < tolerance);
	assert(fabs(m2 - 1.0)  < tolerance);
	assert(fabs(m3 - 0.0)  < tolerance);
	assert(fabs(m4 - 3.0)  < tolerance);
	assert(fabs(m5 - 0.0)  < tolerance);
	assert(fabs(m6 - 10.0) < tolerance);

      }
    }
  }

  free(moment6);
  noise_free(noise);
  cs_free(cs);

  return 0;
}
