/*****************************************************************************
 *
 *  hydro_options.c
 *
 *  Options for hydrodynamics sector.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "hydro_options.h"

/*****************************************************************************
 *
 *  hydro_options_default
 *
 *  At the moment, there's not a great deal to see here.
 *
 *  There is almost never any case for using anything other than nhcomm = 1
 *  in a real computation.
 *
 *****************************************************************************/

hydro_options_t hydro_options_default(void) {

  hydro_options_t opts = {.nhcomm = 1, .haloscheme = HYDRO_U_HALO_TARGET};

  return opts;
}

/*****************************************************************************
 *
 *  hydro_options_nhalo
 *
 *  Useful to set halo width (mostly in tests).
 *
 *****************************************************************************/

hydro_options_t hydro_options_nhalo(int nhalo) {

  hydro_options_t opts = hydro_options_default();

  assert(nhalo >= 0);

  opts.nhcomm = nhalo;

  return opts;
}
