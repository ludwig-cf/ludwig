/*****************************************************************************
 *
 *  noise_options.c
 *
 *  Options for lattice fluctuation generator.
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "noise_options.h"

/*****************************************************************************
 *
 *  noise_options_default
 *
 *****************************************************************************/

noise_options_t noise_options_default(void) {

  noise_options_t opts = {
    .seed = 13,                        /* Historical choice */
    .nextra = 1,                       /* Halo extent for random fluxes */
    .iodata = io_info_args_default(),
    .filestub = "noise"                /* Internal option only */
  };

  return opts;
}

/*****************************************************************************
 *
 *  noise_options_seed
 *
 *  No check at this point, but seed != 0 please.
 *
 *****************************************************************************/

noise_options_t noise_options_seed(unsigned int seed) {

  noise_options_t opts = noise_options_default();

  opts.seed = seed;

  return opts;
}

/*****************************************************************************
 *
 *  noise_options_seed_nextra
 *
 *****************************************************************************/

noise_options_t noise_options_seed_nextra(unsigned int seed, int nextra) {

  noise_options_t opts = noise_options_seed(seed);

  opts.nextra = nextra;

  return opts;
}

/*****************************************************************************
 *
 *  noise_options_valid
 *
 *  Return 1 for valid, 0 for invalid.
 *
 *****************************************************************************/

int noise_options_valid(const noise_options_t * options) {

  int isvalid = 1;

  assert(options);

  if (options == NULL) {
    isvalid = 0;
  }
  else {
    if (options->seed <= 0) isvalid = 0;
    if (0 > options->nextra || options->nextra > 1) isvalid = 0;
    if (io_options_valid(&options->iodata.input) == 0) isvalid = 0;
    if (io_options_valid(&options->iodata.output) == 0) isvalid = 0;
  }

  return isvalid;
}
