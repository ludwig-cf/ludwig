/*****************************************************************************
 *
 *  map_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MAP_OPTIONS_H
#define LUDWIG_MAP_OPTIONS_H

#include "io_info_args.h"

typedef struct map_options_s map_options_t;

struct map_options_s {
  int ndata;
  int is_porous_media;
  const char * filestub;

  io_info_args_t iodata;
};

map_options_t map_options_default(void);
map_options_t map_options_ndata(int ndata);
int map_options_valid(const map_options_t * map);

#endif
