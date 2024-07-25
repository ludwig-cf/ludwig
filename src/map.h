/*****************************************************************************
 *
 *  map.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_MAP_H
#define LUDWIG_MAP_H

#include "pe.h"
#include "coords.h"
#include "memory.h"
#include "io_impl.h"
#include "map_options.h"

#define MAP_DATA_RECORD_LENGTH_ASCII 23

enum map_status {MAP_FLUID, MAP_BOUNDARY, MAP_COLLOID, MAP_STATUS_MAX};

typedef struct map_s map_t;

struct map_s {
  int nsite;                  /* Number of sites allocated */
  int is_porous_media;        /* Flag for porous media */
  int ndata;                  /* Additional fields associated with map */
  char * status;              /* Status (one of enum_status) */
  double * data;              /* Additional site lattice property */

  pe_t * pe;                  /* Parallel environment */
  cs_t * cs;                  /* Coordinate system */

  io_element_t ascii;         /* Per site ascii information */
  io_element_t binary;        /* Per site binary information */
  io_metadata_t input;        /* Information for input */
  io_metadata_t output;       /* Informtation for output */

  map_options_t options;      /* Copy of options */
  const char * filestub;      /* Filename stub for map files usually "map" */

  map_t * target;             /* Copy of this structure on target */
};

int map_create(pe_t * pe, cs_t * cs, const map_options_t * options,
	       map_t ** map);
int map_free(map_t ** map);
int map_initialise(pe_t * pe, cs_t * cs, const map_options_t * options,
		   map_t * map);
int map_finalise(map_t * map);

int map_memcpy(map_t * map, tdpMemcpyKind flag);
int map_pm(map_t * map, int * porous_media_flag);
int map_pm_set(map_t * map, int porous_media_flag);
int map_volume_local(map_t * obj, int status, int * volume);
int map_volume_allreduce(map_t * obj, int status, int * volume);
int map_halo(map_t * obj);

int map_read_buf(map_t * map, int index, const char * buf);
int map_read_buf_ascii(map_t * map, int index, const char * buf);
int map_write_buf(map_t * map, int index, char * buf);
int map_write_buf_ascii(map_t * map, int index, char * buf);
int map_io_aggr_pack(map_t * map, io_aggregator_t * aggr);
int map_io_aggr_unpack(map_t * map, const io_aggregator_t * aggr);
int map_io_read(map_t * map, int timestep);
int map_io_write(map_t * map, int timestep);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions
 *
 *****************************************************************************/

/*****************************************************************************
 *
 *  map_status
 *
 *  Return map status as an integer.
 *
 *****************************************************************************/

__host__ __device__
static inline int map_status(const map_t * map, int index, int * status) {

  assert(map);
  assert(status);

  *status = (int) map->status[addr_rank0(map->nsite, index)];

  return 0;
}

/*****************************************************************************
 *
 *  map_status_set
 *
 *****************************************************************************/

__host__ __device__
static inline int map_status_set(map_t * map, int index, int status) {

  assert(map);
  assert(0 <= status && status < MAP_STATUS_MAX);

  map->status[addr_rank0(map->nsite, index)] = status;

  return 0;
}

/*****************************************************************************
 *
 *  map_data
 *
 *****************************************************************************/

__host__ __device__
static inline int map_data(const map_t * map, int index, double * data) {

  assert(map);
  assert(data);

  for (int n = 0; n < map->ndata; n++) {
    data[n] = map->data[addr_rank1(map->nsite, map->ndata, index, n)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_data_set
 *
 *****************************************************************************/

__host__ __device__
static inline int map_data_set(map_t * map, int index, double * data) {

  assert(map);
  assert(data);

  for (int n = 0; n < map->ndata; n++) {
    map->data[addr_rank1(map->nsite, map->ndata, index, n)] = data[n];
  }

  return 0;
}

#endif
