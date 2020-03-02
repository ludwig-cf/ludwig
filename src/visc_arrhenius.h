/*****************************************************************************
 *
 *  visc_arrhenius.h
 *
 *  Arrhenius viscosity model.
 *  See visc_arrhenius.c for further details.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_VISC_ARRHENIUS_H
#define LUDWIG_VISC_ARRHENIUS_H

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "visc.h"

typedef struct visc_arrhenius_param_s visc_arrhenius_param_t;
typedef struct visc_arrhenius_s visc_arrhenius_t;

/* Viscosity model */

struct visc_arrhenius_s {
  visc_t super;                      /* Superclass block */
  pe_t * pe;                         /* Parallel environment */
  cs_t * cs;                         /* Coordinate system */
  field_t * phi;                     /* Composition field */
  visc_arrhenius_param_t * param;    /* Parameters */
};

/* Parameters */

struct visc_arrhenius_param_s {
  double eta_minus;
  double eta_plus;
  double phistar;
};

__host__ int visc_arrhenius_create(pe_t * pe, cs_t * cs, field_t * phi,
				   visc_arrhenius_param_t param,
				   visc_arrhenius_t ** visc);
__host__ int visc_arrhenius_free(visc_arrhenius_t * visc);
__host__ int visc_arrhenius_info(visc_arrhenius_t * visc);
__host__ int visc_arrhenius_update(visc_arrhenius_t * visc, hydro_t * hydro);
__host__ int visc_arrhenius_stats(visc_arrhenius_t * visc, hydro_t * hydro);

#endif
