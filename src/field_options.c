/*****************************************************************************
 *
 *  field_options.c
 *
 *  Wrangle field options.
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

#include <assert.h>

#include "field_options.h"

/*****************************************************************************
 *
 *  field_options_default
 *
 *****************************************************************************/

field_options_t field_options_default(void) {

  field_options_t opts = {.ndata  = 1,
                          .nhcomm = 0,
                          .haloscheme = FIELD_HALO_TARGET,
                          .iodata = io_info_args_default()};
  return opts;
}

/*****************************************************************************
 *
 *  field_options_ndata_nhalo
 *
 *****************************************************************************/

field_options_t field_options_ndata_nhalo(int ndata, int nhalo) {

  field_options_t opts = field_options_default();

  opts.ndata  = ndata;
  opts.nhcomm = nhalo;

  return opts;
}

/*****************************************************************************
 *
 *  field_options_valid
 *
 *****************************************************************************/

int field_options_valid(const field_options_t * opts) {

  int valid = 1;

  if (opts->ndata  < 1) valid = 0;
  if (opts->nhcomm < 0) valid = 0; /* Must also be < nhalo ... */

  return valid;
}
