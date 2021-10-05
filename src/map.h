/*****************************************************************************
 *
 *  map.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2021 The University of Edinburgh
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
#include "io_harness.h"

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
  MPI_Datatype halostatus[3]; /* Halo datatype for status */
  MPI_Datatype halodata[3];   /* Halo datatype for data */
  io_info_t * info;           /* I/O handler */

  map_t * target;             /* Copy of this structure on target */
};

__host__ int map_create(pe_t * pe, cs_t * cs, int ndata, map_t ** pobj);
__host__ int map_free(map_t * obj);
__host__ int map_memcpy(map_t * map, tdpMemcpyKind flag);

__host__ int map_pm(map_t * map, int * porous_media_flag);
__host__ int map_pm_set(map_t * map, int porous_media_flag);
__host__ int map_volume_local(map_t * obj, int status, int * volume);
__host__ int map_volume_allreduce(map_t * obj, int status, int * volume);
__host__ int map_halo(map_t * obj);
__host__ int map_init_io_info(map_t * obj, int grid[3], int form_in, int form_out);
__host__ int map_io_info(map_t * obj, io_info_t ** info);

__host__ __device__ int map_status(map_t * obj, int index, int * status);
__host__ __device__ int map_status_set(map_t * obj, int index, int status);
__host__ __device__ int map_data(map_t * obj, int index, double * data);
__host__ __device__ int map_data_set(map_t * obj, int index, double * data);
__host__ __device__ int map_ndata(map_t * obj, int * ndata);

#endif
