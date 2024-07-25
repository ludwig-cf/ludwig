/*****************************************************************************
 *
 *  map_options.c
 *
 *  The presence of the wetting data in the map structure may be open
 *  to question: it is here for now.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "map_options.h"

/*****************************************************************************
 *
 *  map_options_default
 *
 *****************************************************************************/

map_options_t map_options_default(void) {

  map_options_t opts = {
    .ndata = 0,                           /* No wetting constants */
    .is_porous_media = 0,                 /* No porous media */
    .filestub = "map",
    .iodata = io_info_args_default()
  };

  return opts;
}

/*****************************************************************************
 *
 *  map_options_ndata
 *
 *****************************************************************************/

map_options_t map_options_ndata(int ndata) {

  /* There are no checks here ... */

  map_options_t opts = map_options_default();
  opts.ndata = ndata;

  return opts;
}

/*****************************************************************************
 *
 *  map_options_valid
 *
 *****************************************************************************/

int map_options_valid(const map_options_t * opts) {

  int valid = 1;

  if (opts == NULL) {
    valid = 0;
  }
  else {
    if (opts->ndata < 0) valid = 0;
    if (opts->ndata > 2) valid = 0; /* At most two wetting constants. */
    if (io_options_valid(&opts->iodata.input) == 0) valid = 0;
    if (io_options_valid(&opts->iodata.output) == 0) valid = 0;
  }

  return valid;
}
