/*****************************************************************************
 *
 *  heat_equation.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2019 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_HEAT_EQUATION_H
#define LUDWIG_HEAT_EQUATION_H

#include "pe.h"
#include "coords.h"
#include "advection.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "map.h"
#include "noise.h"

typedef struct heq_s heq_t;
typedef struct heq_info_s heq_info_t;

struct heq_info_s {
  int conserve; /* 0 = normal; 1 = compensated sum */
};

struct heq_s {
  heq_info_t info;
  pe_t * pe;
  cs_t * cs;
  field_t * csum;
  lees_edw_t * le;
  advflux_t * flux;
};

__host__ int heq_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			   heq_info_t * info,
			   heq_t ** heq);
__host__ int heq_free(heq_t * heq);

__host__ int heat_equation(heq_t * heq, fe_t * fe, field_t * temperature,
			       hydro_t * hydro, map_t * map,
			       noise_t * noise);

#endif
