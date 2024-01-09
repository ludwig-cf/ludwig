/*****************************************************************************
 *
 *  field.h
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

#ifndef LUDWIG_FIELD_H
#define LUDWIG_FIELD_H

#include <stdint.h>

#define NVECTOR 3    /* Storage requirement for vector (per site) */
#define NQAB 5       /* Storage requirement for symmetric, traceless tensor */

#include "pe.h"
#include "coords.h"
#include "io_impl.h"
#include "io_event.h"
#include "io_harness.h"       /* To be removed in favour of refactored io */
#include "kernel.h"
#include "leesedwards.h"
#include "field_options.h"

/* Halo */

#include "cs_limits.h"

typedef struct field_graph_halo_s field_graph_halo_t;

struct field_graph_halo_s {
  tdpGraph_t graph;
  tdpGraphExec_t exec;
};

typedef struct field_halo_s field_halo_t;

struct field_halo_s {

  MPI_Comm comm;                /* coords: Cartesian communicator */
  int nbrrank[3][3][3];         /* coords: Cartesian neighbours */

  int nvel;                     /* Number of directions involved (2d or 3d) */
  int8_t cv[27][3];             /* Send/recv directions */
  cs_limits_t slim[27];         /* halo: send regions (rectangular) */
  cs_limits_t rlim[27];         /* halo: recv regions (rectangular) */
  double * send[27];            /* halo: send data buffers */
  double * recv[27];            /* halo: recv data buffers */
  MPI_Request request[2*27];    /* halo: array of send/recv requests */

  tdpStream_t stream;
  field_halo_t * target;        /* target structure */
  double * send_d[27];          /* halo: device send data buffers */
  double * recv_d[27];          /* halo: device recv data buffers */

  field_graph_halo_t gsend;     /* Graph API halo swap */
  field_graph_halo_t grecv;
};

typedef struct field_s field_t;

struct field_s {
  int nf;                       /* Number of field components */
  int nhcomm;                   /* Halo width required */
  int nsites;                   /* Local sites (allocated) */
  double * data;                /* Field data */
  const char * name;            /* "phi", "p", "q" etc. */

  double field_init_sum;        /* field sum at the beginning */

  pe_t * pe;                    /* Parallel environment */
  cs_t * cs;                    /* Coordinate system */
  lees_edw_t * le;              /* Lees-Edwards */

  io_metadata_t iometadata_in;  /* Input details */
  io_metadata_t iometadata_out; /* Output details */

  io_info_t * info;             /* I/O Handler (to be removed) */

  field_halo_t h;               /* Host halo */
  field_options_t opts;         /* Options */

  field_t * target;             /* target structure */
};


int field_halo_create(const field_t * field, field_halo_t * h);
int field_halo_post(const field_t * field, field_halo_t * h);
int field_halo_wait(field_t * field, field_halo_t * h);
int field_halo_info(const field_t * field);
int field_halo_free(field_halo_t * h);

__host__ int field_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  const char * name,
			  const field_options_t * opts,
			  field_t ** pobj);
__host__ int field_free(field_t * obj);

__host__ int field_memcpy(field_t * obj, tdpMemcpyKind flag);
__host__ int field_init_io_info(field_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int field_io_info(field_t * obj, io_info_t ** info);
__host__ int field_halo(field_t * obj);
__host__ int field_halo_swap(field_t * obj, field_halo_enum_t flag);
__host__ int field_leesedwards(field_t * obj);

__host__ __device__ int field_nf(field_t * obj, int * nop);
__host__ __device__ int field_scalar(field_t * obj, int index, double * phi);
__host__ __device__ int field_scalar_set(field_t * obj, int index, double phi);
__host__ __device__ int field_vector(field_t * obj, int index, double p[3]);
__host__ __device__ int field_vector_set(field_t * obj, int index,
					 const double p[3]);
__host__ __device__ int field_tensor(field_t * obj, int index, double q[3][3]);
__host__ __device__ int field_tensor_set(field_t * obj, int index,
					 double q[3][3]);
__host__ __device__ int field_scalar_array(field_t * obj, int index,
					   double * array);
__host__ __device__ int field_scalar_array_set(field_t * obj, int index,
					       const double * array);

int field_read_buf(field_t * field, int index, const char * buf);
int field_read_buf_ascii(field_t * field, int index, const char * buf);
int field_write_buf(field_t * field, int index, char * buf);
int field_write_buf_ascii(field_t * field, int index, char * buf);
int field_io_aggr_pack(field_t * field, io_aggregator_t * aggr);
int field_io_aggr_unpack(field_t * field, const io_aggregator_t * aggr);

int field_io_write(field_t * field, int timestep, io_event_t * event);
int field_io_read(field_t * field, int timestep, io_event_t * event);

int field_graph_halo_send_create(const field_t * field, field_halo_t * h);
int field_graph_halo_recv_create(const field_t * field, field_halo_t * h);

#endif
