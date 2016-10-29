/*****************************************************************************
 *
 *  wall.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef WALL_H
#define WALL_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "model.h"
#include "map.h"

typedef struct wall_param_s wall_param_t;
typedef struct wall_s wall_t;

struct wall_param_s {
  int iswall;           /* Flag for flat walls */
  int isporousmedia;    /* Flag for porous media */
  int isboundary[3];    /* X, Y, Z boundary markers */
  int initshear;        /* Use shear initialisation of distributions */
  double ubot[3];       /* 'Botttom' wall motion */
  double utop[3];       /* 'Top' wall motion */
  double lubr_rc[3];    /* Lubrication correction cut offs */
};


__host__ int wall_create(pe_t * pe, cs_t * cs, map_t * map, lb_t * lb,
			 wall_t ** p);
__host__ int wall_free(wall_t * wall);
__host__ int wall_info(wall_t * wall);
__host__ int wall_commit(wall_t * wall, wall_param_t values);
__host__ int wall_target(wall_t * wall, wall_t ** target);
__host__ int wall_param(wall_t * wall, wall_param_t * param);
__host__ int wall_param_set(wall_t * wall, wall_param_t values);
__host__ int wall_shear_init(wall_t * wall);

__host__ int wall_bbl(wall_t * wall);
__host__ int wall_present(wall_t * wall);
__host__ int wall_present_dim(wall_t * wall, int iswall[3]);
__host__ int wall_momentum(wall_t * wall, double g[3]);
__host__ int wall_momentum_add(wall_t * wall, const double g[3]);
__host__ int wall_set_wall_distributions(wall_t * wall);
__host__ int wall_lubr_sphere(wall_t * wall,  double ah, const double r[3],
			      double  drag[3]);
__host__ int wall_is_pm(wall_t * wall, int * ispm);

#endif
