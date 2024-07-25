/*****************************************************************************
 *
 *  noise_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_NOISE_OPTIONS_H
#define LUDWIG_NOISE_OPTIONS_H

#include "io_info_args.h"

typedef struct noise_options_s noise_options_t;

struct noise_options_s {
  unsigned int seed;        /* Single overall random seed (!= 0) */
  int nextra;               /* Extent of halo region with need of state */
  io_info_args_t iodata;    /* I/O options */
  const char * filestub;    /* Restart file name stub */
};

noise_options_t noise_options_default(void);
noise_options_t noise_options_seed(unsigned int seed);
noise_options_t noise_options_seed_nextra(unsigned int seed, int nextra);

int noise_options_valid(const noise_options_t * options);

#endif
