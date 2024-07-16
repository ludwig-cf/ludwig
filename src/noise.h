/*****************************************************************************
 *
 *  noise.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_NOISE_H
#define LUDWIG_NOISE_H

#include "pe.h"
#include "coords.h"
#include "io_impl.h"
#include "io_event.h"
#include "noise_options.h"

/* NNOISE_MAX                    10 discrete variates per uniform variate */
/* NNOISE_STATE                  4  int32_t per state */
/* NOISE_RECORD_LENGTH_ASCII     11 characters in ascii record per int32 */

#define NNOISE_MAX 10
#define NNOISE_STATE 4
#define NOISE_RECORD_LENGTH_ASCII 11

typedef struct noise_s noise_t;

struct noise_s {
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */
  int nsites;               /* Total number of lattice sites */
  unsigned int * state;     /* Local state */
  double rtable[8];         /* Look up table following Ladd (2009). */

  noise_options_t options;  /* Options */
  io_element_t ascii;       /* Per site ascii information */
  io_element_t binary;      /* Per site binary format */
  io_metadata_t input;      /* Defines input behaviour */
  io_metadata_t output;     /* Defines output behaviour */

  noise_t * target;         /* Target copy */
};

int noise_create(pe_t * pe, cs_t * cs, const noise_options_t * options,
		   noise_t ** ns);
int noise_create_seed(pe_t * pe, cs_t * cs, int seed, noise_t ** ns);
int noise_free(noise_t ** ns);
int noise_initialise(pe_t * pe, cs_t * cs, const noise_options_t * options,
		     noise_t * ns);
int noise_finalise(noise_t * ns);

int noise_memcpy(noise_t * obj, tdpMemcpyKind flag);
int noise_read_buf(noise_t * ns, int index, const char * buf);
int noise_read_buf_ascii(noise_t * ns, int index, const char * buf);
int noise_write_buf(const noise_t * ns, int index, char * buf);
int noise_write_buf_ascii(const noise_t * ns, int index, char * buf);
int noise_io_aggr_pack(const noise_t * cs, io_aggregator_t * aggr);
int noise_io_aggr_unpack(noise_t * ns, const io_aggregator_t * aggr);
int noise_io_write(noise_t * ns, int timestep, io_event_t * event);
int noise_io_read(noise_t * cs, int timestep, io_event_t * event);

int noise_state_set(noise_t * ns, int index, const unsigned int s[4]);
int noise_state(const noise_t * ns, int index, unsigned int s[4]);

int noise_uniform_double_reap(noise_t * obj, int index, double * reap);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions.
 *
 *****************************************************************************/

#include "memory.h"

/*****************************************************************************
 *
 *  ns_uniform
 *
 *  Return a uniformly distributed integer following Marsaglia 1999.
 *  The range is 0 - 2^32 - 1.
 *
 *  This is the implementation of L'Ecuyer and Simard found in the
 *  testu01 package (v1.2.2)
 *  http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
 *
 *  A "class method".
 *
 *****************************************************************************/

__host__ __device__ static inline unsigned int ns_uniform(unsigned int s[4]) {

  unsigned int b;

  s[0] = 69069*s[0] + 1234567;
  b = s[1] ^ (s[1] << 17);
  b ^= (b >> 13);
  s[1] = b ^ (b << 5);
  s[2] = 36969*(s[2] & 0xffff) + (s[2] >> 16);
  s[3] = 18000*(s[3] & 0xffff) + (s[3] >> 16);
  b = (s[2] << 16) + s[3];

  return (s[1] + (s[0] ^ b));
}

/*****************************************************************************
 *
 *  noise_discrete_m10
 *
 *  Return a multiple of NNOISE_MAX discrete random numbers.
 *
 *****************************************************************************/

__host__ __device__ static inline void noise_discrete_m10(noise_t * ns,
							  int index,
							  int mult,
							  double * reap) {
  unsigned int state[NNOISE_STATE] = {0};

  assert(ns);
  assert(0 <= index && index < ns->nsites);
  assert(mult == 1 || mult == 2); /* Expected */
  assert(reap);

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    state[ia] = ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)];
  }

  for (int m = 0; m < mult; m++) {

    /* Remove the leading two bits, and index the table using each of the
     * remaining three bits in turn. */

    unsigned int iuniform = ns_uniform(state);
    iuniform >>= 2;

    for (int ia = 0; ia < NNOISE_MAX; ia++) {
      reap[m*NNOISE_MAX + ia] = ns->rtable[iuniform & 7];
      iuniform >>= 3;
    }
  }

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)] = state[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  noise_reap_n
 *
 *  Return nmax discrete random numbers for site index.
 *  These have mean zero and variance unity.
 *
 *****************************************************************************/

__host__ __device__ static inline void noise_reap_n(noise_t * ns,
						    int index,
						    int nmax,
						    double * reap) {
  unsigned int iuniform;
  unsigned int state[NNOISE_STATE];

  assert(ns);
  assert(0 <= index && index < ns->nsites);
  assert(nmax <= NNOISE_MAX);

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    state[ia] = ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)];
  }

  /* Remove the leading two bits, and index the table using each of the
   * remaining three bits in turn. */

  iuniform = ns_uniform(state);
  iuniform >>= 2;

  for (int ia = 0; ia < nmax; ia++) {
    reap[ia] = ns->rtable[iuniform & 7];
    iuniform >>= 3;
  }

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)] = state[ia];
  }

  return;
}

#endif
