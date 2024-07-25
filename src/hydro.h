/*****************************************************************************
 *
 *  hydro.h
 *
 *  Hydrodynamic sector: rho, velocity, force, viscosity.
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

#ifndef LUDWIG_HYDRO_H
#define LUDWIG_HYDRO_H

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "leesedwards.h"
#include "hydro_options.h"

typedef struct hydro_s hydro_t;

/* Data storage: Always a 3-vector NHDIM */

#define NHDIM 3

struct hydro_s {

  int nsite;               /* Allocated sites (local) */
  int nhcomm;              /* Width of halo region for u field */

  field_t * rho;           /* Density field */
  field_t * u;             /* velocity field */
  field_t * force;         /* Body force on fluid */
  field_t * eta;           /* Scalar viscosity field */

  pe_t * pe;               /* Parallel environment */
  cs_t * cs;               /* Coordinate system */
  lees_edw_t * le;         /* Lees Edwards */

  hydro_t * target;        /* structure on target */
};

__host__ int hydro_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  const hydro_options_t * opts, hydro_t ** pobj);
__host__ int hydro_free(hydro_t * obj);
__host__ int hydro_memcpy(hydro_t * ibj, tdpMemcpyKind flag);
__host__ int hydro_u_halo(hydro_t * obj);
__host__ int hydro_halo_swap(hydro_t * obj, field_halo_enum_t flag);
__host__ int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
				     double w[3][3]);
__host__ int hydro_lees_edwards(hydro_t * obj);
__host__ int hydro_correct_momentum(hydro_t * obj);
__host__ int hydro_f_zero(hydro_t * obj, const double fzero[3]);
__host__ int hydro_u_zero(hydro_t * obj, const double uzero[3]);
__host__ int hydro_rho0(hydro_t * hydro, double rho0);

__host__ int hydro_io_write(hydro_t * hydro, int timestep, io_event_t * event);
__host__ int hydro_io_read(hydro_t * hydro, int timestep, io_event_t * event);

#include "hydro_impl.h"

#endif
