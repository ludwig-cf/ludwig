/*****************************************************************************
 *
 *  lc_droplet.h
 *
 *  Routines related to liquid crystal droplet free energy
 *  and molecular field.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Juho Lintuvuori
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LC_DROPLET_H
#define LUDWIG_LC_DROPLET_H

#include "coords.h"
#include "blue_phase.h"
#include "symmetric.h"
#include "field.h"
#include "field_grad.h"
#include "hydro.h"

typedef struct fe_lc_droplet_s fe_lc_droplet_t;
typedef struct fe_lc_droplet_param_s fe_lc_droplet_param_t;

struct fe_lc_droplet_s {
  fe_t super;
  pe_t * pe;                      /* Parallel environment */
  cs_t * cs;                      /* Coordinate system */
  fe_lc_droplet_param_t * param;  /* Coupling parameters */
  fe_lc_t * lc;                   /* LC free energy  etc */
  fe_symm_t * symm;               /* Symmetric free energy etc */
  fe_lc_droplet_t * target;       /* Target pointer */
};

struct fe_lc_droplet_param_s {
  double gamma0; /* \gamma(phi) = gamma0 + delta x (1 + phi) */
  double delta;  /* As above */
  double w;      /* Surface nchoring constant */
};

__host__ int fe_lc_droplet_create(pe_t * pe, cs_t * cs, fe_lc_t * lc,
				  fe_symm_t * symm,
				  fe_lc_droplet_t ** p);
__host__ int fe_lc_droplet_free(fe_lc_droplet_t * fe);
__host__ int fe_lc_droplet_param(fe_lc_droplet_t * fe,
				 fe_lc_droplet_param_t * param);
__host__ int fe_lc_droplet_param_set(fe_lc_droplet_t * fe,
				     fe_lc_droplet_param_t param);
__host__ int fe_lc_droplet_target(fe_lc_droplet_t * fe, fe_t ** target);
__host__ int fe_lc_droplet_bodyforce(fe_lc_droplet_t * fe, hydro_t * hydro);

__host__ __device__
int fe_lc_droplet_gamma(fe_lc_droplet_t * fe, int index,  double * gamma);

__host__ __device__ int fe_lc_droplet_fed(fe_lc_droplet_t * fe, int index,
					  double * fed);
__host__ __device__ int fe_lc_droplet_stress(fe_lc_droplet_t * fe, int index,
					     double s[3][3]);
__host__ __device__ void fe_lc_droplet_stress_v(fe_lc_droplet_t * fe,
						int index,
					       double s[3][3][NSIMDVL]);
__host__ __device__ int fe_lc_droplet_mol_field(fe_lc_droplet_t * fe,
						int index,
						double h[3][3]);
__host__ __device__ void fe_lc_droplet_mol_field_v(fe_lc_droplet_t * fe,
						   int index,
						   double h[3][3][NSIMDVL]);
__host__ __device__ int fe_lc_droplet_mu(fe_lc_droplet_t * fe, int index,
					 double * mu);

__host__ __device__ void fe_lc_droplet_stress_v(fe_lc_droplet_t * fe,
						int index,
						double sth[3][3][NSIMDVL]);
__host__ __device__ void fe_lc_droplet_mol_field_v(fe_lc_droplet_t * fe,
						   int index,
						   double h[3][3][NSIMDVL]);

#endif
