/*****************************************************************************
 *
 *  fluctuations.c
 *
 *  This is to generatate random numbers used for isothermal fluctuations
 *  (aka 'noise'). The idea here is to make this (potentially)
 *  decomposition-independent by allowing each lattice site to
 *  retain its own random number generator state.
 *
 *  The final Gaussian deviates use the discrete generator described
 *  by Ladd in Computer Physics Communications 180, 2140--2142 (2009).
 *
 *  Here, the uniform random numbers are generated via an RNG proposed
 *  by Marsaglia (unpublished, 1999);  It is referred to as KISS99 in
 *  L'Ecuyer and Simard in ACM TOMS 33 Article 22 (2007). The state is
 *  4 4-byte unsigned integers.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 Kevin Stratford
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "fluctuations.h"

/* The implementation is based on the following opaque object, which
 * holds the uniform random number generator state for all sites
 * (here 4*4 byte integer). It also holds a table of the discrete
 * values. */

struct fluctuations_type {
  void * pe_t;              /* Placeholder for pe_t */
  int nsites;
  unsigned int * state;
  double rtable[8];
};

/*****************************************************************************
 *
 *  fluctuations_create
 *
 *  We allocate the state for the appropriate number of sites, but set
 *  all the state to zero.
 *
 *****************************************************************************/

fluctuations_t * fluctuations_create(const int nsites) {

  fluctuations_t * f;

  assert(nsites > 0);

  f = (fluctuations_t *) malloc(sizeof(fluctuations_t));
  if (f == NULL) fatal("malloc(fluctuations_t) failed\n");

  f->nsites = nsites;
  f->state = (unsigned int *) calloc(NFLUCTUATION_STATE*nsites,
				     sizeof(unsigned int));
  if (f->state == NULL) fatal("malloc(f->state) failed\n");

  f->rtable[0] = -sqrt(2.0 + sqrt(2.0));
  f->rtable[1] = -sqrt(2.0 - sqrt(2.0));
  f->rtable[2] = 0.0;
  f->rtable[3] = 0.0;
  f->rtable[4] = 0.0;
  f->rtable[5] = 0.0;
  f->rtable[6] = +sqrt(2.0 - sqrt(2.0));
  f->rtable[7] = +sqrt(2.0 + sqrt(2.0));

  return f;
}

/*****************************************************************************
 *
 *  fluctuations_destroy
 *
 *****************************************************************************/

void fluctuations_destroy(fluctuations_t * f) {

  assert(f);
  free(f->state);
  free(f);

  return;
}

/*****************************************************************************
 *
 *  fluctuations_state_set
 *
 *****************************************************************************/

void fluctuations_state_set(fluctuations_t * f, const int index,
			    const unsigned int newstate[NFLUCTUATION_STATE]) {
  int ia;

  assert(f);
  assert(index >= 0);
  assert(index < f->nsites);

  for (ia = 0; ia < NFLUCTUATION_STATE; ia++) {
    assert(newstate[ia] >= 0);
    f->state[NFLUCTUATION_STATE*index + ia] = newstate[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  fluctuations_state
 *
 *  Return the state of the generator at index.
 *
 *****************************************************************************/

void fluctuations_state(const fluctuations_t * f, const int index,
			unsigned int state[NFLUCTUATION_STATE]) {
  int ia;

  assert(f);
  assert(index >= 0);
  assert(index < f->nsites);

  for (ia = 0; ia < NFLUCTUATION_STATE; ia++) {
    state[ia] = f->state[NFLUCTUATION_STATE*index + ia];
  }

  return;
}

/*****************************************************************************
 *
 *  fluctuations_reap
 *
 *  Return NFLUCTAUTION discrete random numbers for site index.
 *
 *****************************************************************************/

void fluctuations_reap(fluctuations_t * f, const int index, double * reap) {

  unsigned int iuniform;
  int ia;

  assert(f);
  assert(index >= 0);
  assert(index < f->nsites);

  iuniform = fluctuations_uniform(f->state + NFLUCTUATION_STATE*index);

  /* Remove the leading two bits, and index the table using each of the
   * remaining three bits in turn. */

  iuniform >>= 2;

  for (ia = 0; ia < NFLUCTUATION; ia++) {
    reap[ia] = f->rtable[iuniform & 7];
    iuniform >>= 3;
  }

  return;
}

/*****************************************************************************
 *
 *  fluctuations_uniform
 *
 *  Return a uniformly distributed integer following Marsaglia 1999.
 *  The range is 0 - 2^32 - 1.
 *
 *  This implementation is a direct rip-off of (ahem, `based upon') the
 *  implementation of L'Ecuyer and Simard found in the
 *  testu01 package (v1.2.2)
 *  http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
 *
 *****************************************************************************/

unsigned int fluctuations_uniform(unsigned int state[NFLUCTUATION_STATE]) {

  unsigned int b;

  state[0] = 69069*state[0] + 1234567;
  b = state[1] ^ (state[1] << 17);
  b ^= (b >> 13);
  state[1] = b ^ (b << 5);
  state[2] = 36969*(state[2] & 0xffff) + (state[2] >> 16);
  state[3] = 18000*(state[3] & 0xffff) + (state[3] >> 16);
  b = (state[2] << 16) + state[3];

  return (state[1] + (state[0] ^ b));
}
