/*****************************************************************************
 *
 *  lb_data_options.c
 *
 *  Options available for distribution data at run time.
 *
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

#include "lb_data_options.h"

/*****************************************************************************
 *
 *  lb_data_options_default
 *
 *****************************************************************************/

lb_data_options_t lb_data_options_default(void) {

  lb_data_options_t opts = {.ndim = 3, .nvel = 19, .ndist = 1,
                            .nrelax = LB_RELAXATION_M10,
			    .halo   = LB_HALO_TARGET,
			    .reportimbalance = 0,
                            .data = io_info_args_default(),
                            .rho  = io_info_args_default()};

  return opts;
}

/*****************************************************************************
 *
 *  lb_data_options_valid
 *
 *****************************************************************************/

int lb_data_options_valid(const lb_data_options_t * opts) {

  int valid = 1;

  if (!(opts->ndim  == 2 || opts->ndim  == 3)) valid = 0;
  if (!(opts->ndist == 1 || opts->ndist == 2)) valid = 0;

  if (opts->ndist == 2 && opts->halo != LB_HALO_TARGET) valid = 0;

  return valid;
}
