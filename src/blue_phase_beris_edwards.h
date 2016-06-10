/*****************************************************************************
 *
 *  blue_phase_beris_edwards.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2009-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_BERIS_EDWARDS_H
#define BLUE_PHASE_BERIS_EDWARDS_H

#include "hydro.h"
#include "field.h"
#include "map.h"
#include "noise.h"

typedef struct beris_edw_s beris_edw_t;
typedef struct beris_edw_param_s beris_edw_param_t;

struct beris_edw_param_s {
  double gamma;  /* Rotational diffusion constant */
};

__host__ int beris_edw_create(beris_edw_t ** pobj);
__host__ int beris_edw_free(beris_edw_t * be);
__host__ int beris_edw_memcpy(beris_edw_t * be, int flag);
__host__ int beris_edw_param_set(beris_edw_t * be, beris_edw_param_t values);

__host__ int beris_edw_update(beris_edw_t * be, field_t * fq,
			      field_grad_t * fq_grad, hydro_t * hydro,
			      map_t * map, noise_t * noise);

__host__ __device__ int beris_edw_tmatrix_set(double t[3][3][NQAB]);

#endif
