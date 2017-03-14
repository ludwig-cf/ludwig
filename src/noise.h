/*****************************************************************************
 *
 *  noise.h
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

#ifndef LUDWIG_NOISE_H
#define LUDWIG_NOISE_H

#define NNOISE_MAX 10
#define NNOISE_STATE 4

#include "pe.h"
#include "coords.h"
#include "io_harness.h"

typedef enum {NOISE_RHO = 0,
	      NOISE_PHI,
	      NOISE_QAB,
	      NOISE_END}
  noise_enum_t;

typedef struct noise_s noise_t;

__host__ int noise_create(pe_t * pe, cs_t * cs, noise_t ** pobj);
__host__ int noise_free(noise_t * obj);
__host__ int noise_init(noise_t * obj, int master_seed);
__host__ int noise_memcpy(noise_t * obj, tdpMemcpyKind flag);
__host__ int noise_target(noise_t * nosie, noise_t ** target);
__host__ int noise_present_set(noise_t * obj, noise_enum_t type, int present);
__host__ int noise_init_io_info(noise_t * obj, int grid[3], int form_in, int form_out);

__host__ __device__ int noise_state_set(noise_t * obj, int index, unsigned int s[NNOISE_STATE]);
__host__ __device__ int noise_state(noise_t * obj, int index, unsigned int s[NNOISE_STATE]);
__host__ __device__ int noise_reap(noise_t * obj, int index, double * reap);
__host__ __device__ int noise_reap_n(noise_t *obj, int index, int nmax, double * reap);
__host__ __device__ int noise_uniform_double_reap(noise_t * obj, int index, double * reap);

__host__ __device__ int noise_present(noise_t * obj, noise_enum_t type, int * present);
__host__ __device__ unsigned int noise_uniform(unsigned int state[NNOISE_STATE]);

#endif
