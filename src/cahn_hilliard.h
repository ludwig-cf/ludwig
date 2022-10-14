/*****************************************************************************
 *
 *  cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_CAHN_HILLIARD_H
#define LUDWIG_CAHN_HILLIARD_H

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "advection.h"
#include "map.h"

typedef struct ch_s ch_t;
typedef struct ch_info_s ch_info_t;

struct ch_info_s {
  int nfield;             /* Actual number of order parameters */
  double mobility[NQAB];  /* Mobilities for maximum NQAB order parameters */
  double grad_mu_phi[3];
  double grad_mu_psi[3];
};

struct ch_s {
  pe_t * pe;
  cs_t * cs;
  advflux_t * flux;
  ch_info_t * info;
  ch_t * target;
};

__host__ int ch_create(pe_t * pe, cs_t * cs, ch_info_t info, ch_t ** ch);
__host__ int ch_free(ch_t * ch);
__host__ int ch_info(ch_t * ch);
__host__ int ch_info_set(ch_t * ch, ch_info_t info);
__host__ int ch_solver(ch_t * ch, fe_t * fe, field_t * phi, hydro_t * hydro,
		       map_t * map, field_t * subgrid_potential, field_t * flux_mask, rt_t * rt);

#endif
