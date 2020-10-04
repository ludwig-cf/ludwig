/*****************************************************************************
 *
 *  wall.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_WALL_H
#define LUDWIG_WALL_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "model.h"
#include "map.h"

typedef enum wall_slip_enum {WALL_NO_SLIP = 0,
			     WALL_SLIP_XBOT, WALL_SLIP_XTOP,
			     WALL_SLIP_YBOT, WALL_SLIP_YTOP,
			     WALL_SLIP_ZBOT, WALL_SLIP_ZTOP,
			     WALL_SLIP_MAX} wall_slip_enum_t;

typedef struct wall_slip_s  wall_slip_t;
typedef struct wall_param_s wall_param_t;
typedef struct wall_s       wall_t;

struct wall_slip_s {
  int active;              /* slip conditions required at all? */
  double s[WALL_SLIP_MAX]; /* Table for fraction of slip per wall */
};

struct wall_param_s {
  int iswall;              /* Flag for flat walls */
  int isporousmedia;       /* Flag for porous media */
  int isboundary[3];       /* X, Y, Z boundary markers */
  int initshear;           /* Use shear initialisation of distributions */
  double ubot[3];          /* 'Botttom' wall motion */
  double utop[3];          /* 'Top' wall motion */
  double lubr_rc[3];       /* Lubrication correction cut offs */
  wall_slip_t slip;        /* Slip parameters */
};

struct wall_s {
  pe_t * pe;             /* Parallel environment */
  cs_t * cs;             /* Reference to coordinate system */
  map_t * map;           /* Reference to map structure */
  lb_t * lb;             /* Reference to LB information */ 
  wall_t * target;       /* Device memory */

  wall_param_t * param;  /* parameters */
  int   nlink;           /* Number of links */
  int * linki;           /* outside (fluid) site indices */
  int * linkj;           /* inside (solid) site indices */
  int * linkp;           /* LB basis vectors for links */
  int * linku;           /* Link wall_uw_enum_t (wall velocity) */
  double fnet[3];        /* Momentum accounting for source/sink walls */

  int * linkk;           /* slip complementary fluid site index */
  int8_t * linkq;        /* slip complementary link index (out from k) */
  int8_t * links;        /* Fraction of slip (0 = no slip, 1 = free slip) */
};

__host__ wall_slip_t wall_slip(double sbot[3], double stop[3]);
__host__ int         wall_slip_valid(const wall_slip_t * ws);


__host__ int wall_create(pe_t * pe, cs_t * cs, map_t * map, lb_t * lb,
			 wall_t ** p);
__host__ int wall_free(wall_t * wall);
__host__ int wall_info(wall_t * wall);
__host__ int wall_commit(wall_t * wall, wall_param_t * values);
__host__ int wall_target(wall_t * wall, wall_t ** target);
__host__ int wall_param(wall_t * wall, wall_param_t * param);
__host__ int wall_param_set(wall_t * wall, wall_param_t * values);
__host__ int wall_shear_init(wall_t * wall);
__host__ int wall_memcpy(wall_t * wall, tdpMemcpyKind flag);

__host__ int wall_bbl(wall_t * wall);
__host__ int wall_set_wall_distributions(wall_t * wall);
__host__ int wall_lubr_sphere(wall_t * wall,  double ah, const double r[3],
			      double  drag[3]);
__host__ int wall_momentum(wall_t * wall, double g[3]);
__host__ int wall_momentum_add(wall_t * wall, const double g[3]);

__host__ __device__ int wall_is_pm(wall_t * wall, int * ispm);
__host__ __device__ int wall_present(wall_t * wall);
__host__ __device__ int wall_present_dim(wall_t * wall, int iswall[3]);

#endif
