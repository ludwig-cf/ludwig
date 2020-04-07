/*****************************************************************************
 *
 *  visc.h
 *
 *  Viscosity model (abstract).
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

#ifndef LUDWIG_VISC_H
#define LUDWIG_VISC_H

#include "hydro.h"

typedef enum {VISC_MODEL_ARRHENIUS} visc_model_enum_t;

typedef struct visc_vt_s visc_vt_t;
typedef struct visc_s visc_t;

typedef int (* visc_free_ft)   (visc_t * visc);
typedef int (* visc_update_ft) (visc_t * visc, hydro_t * hydro);
typedef int (* visc_stats_ft)  (visc_t * visc, hydro_t * hydro); 

struct visc_vt_s {
  visc_free_ft   free;      /* Desctructor */
  visc_update_ft update;    /* Update viscosity */
  visc_stats_ft  stats;     /* Viscosity information */
};

struct visc_s {
  const visc_vt_t * func;   /* function table */
  int               id;     /* visc_model_enum_t */
};

#endif

