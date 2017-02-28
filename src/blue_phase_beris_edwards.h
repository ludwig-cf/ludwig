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
 *  (c) 2009-2017 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_BERIS_EDWARDS_H
#define BLUE_PHASE_BERIS_EDWARDS_H

#include "coords.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "hydro.h"
#include "field.h"
#include "map.h"
#include "colloids.h"
#include "noise.h"

typedef struct beris_edw_s beris_edw_t;
typedef struct beris_edw_param_s beris_edw_param_t;

struct beris_edw_param_s {
  double xi;     /* Effective aspect ratio (from relevant free energy) */
  double gamma;  /* Rotational diffusion constant */
  double var;    /* Noise variance */

  double tmatrix[3][3][NQAB];  /* Constant noise tensor */
};

__host__ int beris_edw_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			      beris_edw_t ** pobj);
__host__ int beris_edw_free(beris_edw_t * be);
__host__ int beris_edw_memcpy(beris_edw_t * be, int flag);
__host__ int beris_edw_param_set(beris_edw_t * be, beris_edw_param_t values);
__host__ int beris_edw_param_commit(beris_edw_t * be);

__host__ int beris_edw_update(beris_edw_t * be, fe_t * fe, field_t * fq,
			      field_grad_t * fq_grad, hydro_t * hydro,
			      colloids_info_t * cinfo,
			      map_t * map, noise_t * noise);

__host__ __device__ int beris_edw_tmatrix(double t[3][3][NQAB]);

#endif
