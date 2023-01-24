/*****************************************************************************
 *
 *  hydro_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_HYDRO_OPTIONS_H
#define LUDWIG_HYDRO_OPTIONS_H

#include "field_options.h"

typedef struct hydro_options_s hydro_options_t;

struct hydro_options_s {

  int nhcomm;                        /* Actual halo width */

  field_options_t rho;               /* Density field (scalar) */
  field_options_t u;                 /* Velocity field (vector) */
  field_options_t force;             /* Body force density on fluid (vector) */
  field_options_t eta;               /* Viscosity field (scalar) */
};

hydro_options_t hydro_options_default(void);
hydro_options_t hydro_options_nhalo(int nhalo);
hydro_options_t hydro_options_haloscheme(field_halo_enum_t hs);

#endif
