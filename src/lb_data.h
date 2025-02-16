/*****************************************************************************
 *
 *  lb_data.h
 *
 *  LB distribution data structure implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LB_DATA_H
#define LB_DATA_H

#include <stdint.h>

#include "pe.h"
#include "coords.h"
#include "lb_data_options.h"
#include "lb_model.h"

#include "io_impl.h"
#include "io_event.h"

/* Residual compile-time switches scheduled for removal */
#ifdef _D2Q9_
enum {NDIM = 2, NVEL =  9};
#endif
#ifdef _D3Q15_
enum {NDIM = 3, NVEL = 15};
#endif
#ifdef _D3Q19_
enum {NDIM = 3, NVEL = 19};
#endif
#ifdef _D3Q27_
enum {NDIM = 3, NVEL = 27};
#endif

#define NVELMAX 27
#define LB_RECORD_LENGTH_ASCII 23

typedef struct lb_collide_param_s lb_collide_param_t;
typedef struct lb_halo_s lb_halo_t;
typedef struct lb_data_s lb_t;

typedef struct lb_graph_halo_s lb_graph_halo_t;

struct lb_graph_halo_s {
  tdpGraph_t graph;
  tdpGraphExec_t exec;
};


struct lb_collide_param_s {
  int8_t isghost;                      /* switch for ghost modes */
  int8_t cv[27][3];
  int8_t noise;                        /* switch */
  int nsite;
  int ndist;
  int nvel;
  double rho0;
  double eta_shear;
  double var_shear;
  double eta_bulk;
  double var_bulk;
  double rna[27];                    /* reciprocal of normaliser[p] */
  double rtau[27];
  double wv[27];
  double ma[27][27];
  double mi[27][27];
};

/* Halo */

#include "cs_limits.h"

struct lb_halo_s {

  MPI_Comm comm;                  /* coords: Cartesian communicator */
  int nbrrank[3][3][3];           /* coords: neighbour rank look-up */
  int nlocal[3];                  /* coords: local domain size */

  lb_model_t map;                 /* Communication map 2d or 3d */
  int tagbase;                    /* send/recv tag */
  int full;                       /* All velocities at each site required. */
  int count[27];                  /* halo: item data count per direction */
  cs_limits_t slim[27];           /* halo: send data region (rectangular) */
  cs_limits_t rlim[27];           /* halo: recv data region (rectangular) */
  double * send[27];              /* halo: send buffer per direction */
  double * recv[27];              /* halo: recv buffer per direction */
  MPI_Request request[2*27];      /* halo: array of requests */

  tdpStream_t stream;
  lb_halo_t * target;
  double * send_d[27];            /* halo: device send buffer per direction */
  double * recv_d[27];            /* halo: device recv buffer per direction */

  lb_graph_halo_t gsend;          /* Graph API halo swap */
  lb_graph_halo_t grecv;
};

struct lb_data_s {

  int ndim;
  int nvel;
  int ndist;             /* Number of distributions (default one) */
  int nsite;             /* Number of lattice sites (local) */

  pe_t * pe;             /* parallel environment */
  cs_t * cs;             /* coordinate system */

  lb_model_t model;      /* Current LB model information */

  io_element_t ascii;    /* Per site ASCII information. */
  io_element_t binary;   /* Per site binary information. */
  io_metadata_t input;   /* Metadata for io implementation (input) */
  io_metadata_t output;  /* Ditto (for output) */

  double * f;            /* Distributions */
  double * fprime;       /* used in propagation only */

  lb_collide_param_t * param;   /* Collision parameters REFACTOR THIS */
  lb_relaxation_enum_t nrelax;  /* Relaxation scheme */
  lb_halo_enum_t haloscheme;    /* halo scheme */

  lb_data_options_t opts;       /* Copy of run time options */
  lb_halo_t h;                  /* halo information/buffers */

  double * sbuff;               /* Lees-Edwards crossing buffer "send" */
  double * rbuff;               /* Lees-Edwards crossing buffer "recv" */

  lb_t * target;                /* copy of this structure on target */
};

/* Data storage: A rank two object */

#include "memory.h"

#define LB_ADDR(nsites, ndist, nvel, index, n, p) \
  addr_rank2(nsites, ndist, nvel, index, n, p)

/* Number of hydrodynamic modes */
enum {NHYDRO = 1 + NDIM + NDIM*(NDIM+1)/2};

/* Labels to locate relaxation times in array[NVEL] */
/* Bulk viscosity is XX in stress */
/* Shear is XY in stress */

enum {LB_TAU_BULK = 1 + NDIM + XX, LB_TAU_SHEAR = 1 + NDIM + XY};

#define LB_CS2_DOUBLE(cs2)   const double cs2 = (1.0/3.0)
#define LB_RCS2_DOUBLE(rcs2) const double rcs2 = 3.0

typedef enum lb_dist_enum_type{LB_RHO = 0, LB_PHI = 1} lb_dist_enum_t;
typedef enum lb_mode_enum_type{LB_GHOST_ON = 0, LB_GHOST_OFF = 1} lb_mode_enum_t;

int lb_data_create(pe_t * pe, cs_t * cs,
			    const lb_data_options_t * opts, lb_t ** lb);
int lb_free(lb_t * lb);
int lb_memcpy(lb_t * lb, tdpMemcpyKind flag);
int lb_collide_param_commit(lb_t * lb);
int lb_halo(lb_t * lb);

int lb_init_rest_f(lb_t * lb, double rho0);
int lb_2nd_moment(lb_t * lb, int index, lb_dist_enum_t nd, double s[3][3]);
int lb_1st_moment_equilib_set(lb_t * lb, int index, double rho, double u[3]);

int lb_read_buf(lb_t * lb, int index, const char * buf);
int lb_read_buf_ascii(lb_t * lb, int index, const char * buf);
int lb_write_buf(const lb_t * lb, int index, char * buf);
int lb_write_buf_ascii(const lb_t * lb, int index, char * buf);

int lb_io_aggr_pack(const lb_t * lb, io_aggregator_t * aggr);
int lb_io_aggr_unpack(lb_t * lb, const io_aggregator_t * aggr);

int lb_io_write(lb_t * lb, int timestep, io_event_t * event);
int lb_io_read(lb_t * lb, int timestep, io_event_t * event);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functionm
 *
 *****************************************************************************/

/*****************************************************************************
 *
 *  lb_f
 *
 *  Get the distribution at site index, velocity p, distribution n.
 *
 *****************************************************************************/

__host__ __device__ static inline int lb_f(lb_t * lb, int index, int p, int n,
					   double * f) {

  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(p >= 0 && p < lb->nvel);
  assert(n >= 0 && n < lb->ndist);

  *f = lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p)];

  return 0;
}

/*****************************************************************************
 *
 *  lb_f_set
 *
 *  Set the distribution for site index, velocity p, distribution n.
 *
 *****************************************************************************/

__host__ __device__ static inline int lb_f_set(lb_t * lb, int index, int p,
					       int n, double fvalue) {
  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert(p >= 0 && p < lb->nvel);
  assert(n >= 0 && n < lb->ndist);

  lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p)] = fvalue;

  return 0;
}

/*****************************************************************************
 *
 *  lb_0th_moment
 *
 *  Return the zeroth moment of the distribution (rho for n = 0).
 *
 *****************************************************************************/

__host__ __device__ static inline int lb_0th_moment(lb_t * lb, int index,
						    lb_dist_enum_t nd,
						    double * rho) {
  assert(lb);
  assert(rho);
  assert(index >= 0 && index < lb->nsite);
  assert((int) nd < lb->ndist);

  *rho = 0.0;

  for (int p = 0; p < lb->nvel; p++) {
    *rho += lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, nd, p)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_1st_moment
 *
 *  Return the first moment of the distribution p.
 *
 *****************************************************************************/

__host__ __device__ static inline int lb_1st_moment(lb_t * lb, int index,
						    lb_dist_enum_t nd,
						    double g[3]) {
  assert(lb);
  assert(index >= 0 && index < lb->nsite);
  assert((int) nd < lb->ndist);

  /* Loop to 3 here to cover initialisation in D2Q9 (appears in momentum) */
  for (int n = 0; n < 3; n++) {
    g[n] = 0.0;
  }

  for (int p = 0; p < lb->model.nvel; p++) {
    for (int n = 0; n < lb->model.ndim; n++) {
      g[n] += lb->model.cv[p][n]
	*lb->f[LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, nd, p)];
    }
  }

  return 0;
}

#endif
