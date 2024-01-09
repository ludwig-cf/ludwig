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

  int nhcomm = 1;
  field_options_t rho = field_options_ndata_nhalo(1, nhcomm);
  field_options_t u   = field_options_ndata_nhalo(3, nhcomm);
  field_options_t f   = field_options_ndata_nhalo(3, nhcomm);
  field_options_t eta = field_options_ndata_nhalo(1, nhcomm);

  hydro_options_t opts = {.nhcomm = 1,
                          .rho = rho,
                          .u   = u,
                          .force = f,
                          .eta = eta};
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

  assert(nhalo >= 0);

  field_options_t rho = field_options_ndata_nhalo(1, nhalo);
  field_options_t u   = field_options_ndata_nhalo(3, nhalo);
  field_options_t f   = field_options_ndata_nhalo(3, nhalo);
  field_options_t eta = field_options_ndata_nhalo(1, nhalo);

  hydro_options_t opts = {.nhcomm = nhalo,
                          .rho = rho,
                          .u   = u,
                          .force = f,
                          .eta = eta};

  return opts;
}

/*****************************************************************************
 *
 *  hydro_options_haloscheme
 *
 *****************************************************************************/

hydro_options_t hydro_options_haloscheme(field_halo_enum_t haloscheme) {

  hydro_options_t opts = hydro_options_default();

  opts.rho.haloscheme   = haloscheme;
  opts.u.haloscheme     = haloscheme;
  opts.force.haloscheme = haloscheme;
  opts.eta.haloscheme   = haloscheme;

  return opts;
}
