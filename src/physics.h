/*****************************************************************************
 *
 *  physics.h
 *
 *  $Id: physics.h,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_H
#define PHYSICS_H

#include "pe.h"

typedef struct physics_s physics_t;

__host__ int physics_create(pe_t * pe, physics_t ** phys);
__host__ int physics_free(physics_t * phys);

__host__ int physics_rho0_set(physics_t * phys, double rho0);
__host__ int physics_phi0_set(physics_t * phys, double phi0);
__host__ int physics_eta_shear_set(physics_t * phys, double eta);
__host__ int physics_eta_bulk_set(physics_t * phys, double zeta);
__host__ int physics_kt_set(physics_t * phys, double kt);
__host__ int physics_b0_set(physics_t * phys, double b0[3]);
__host__ int physics_e0_set(physics_t * phys, double e0[3]);
__host__ int physics_e0_frequency_set(physics_t * phys, double e0_frequency);
__host__ int physics_fbody_set(physics_t * phys, double f[3]);
__host__ int physics_fgrav_set(physics_t * phys, double g[3]);
__host__ int physics_mobility_set(physics_t * phys, double mobility);
__host__ int physics_control_next_step(physics_t * phys);
__host__ int physics_control_init_time(physics_t * phys, int nstart, int nstep);
__host__ int physics_fpulse_set(physics_t * phys, double fpulse[3]);
__host__ int physics_fpulse_frequency_set(physics_t * phys, double fpulse_freq);

__host__ __device__ int physics_ref(physics_t ** ref);
__host__ __device__ int physics_rho0(physics_t * phys, double * rho);
__host__ __device__ int physics_phi0(physics_t * phys, double * phi0);
__host__ __device__ int physics_eta_shear(physics_t * phys, double * eta);
__host__ __device__ int physics_eta_bulk(physics_t * phys, double * zeta);
__host__ __device__ int physics_kt(physics_t * phys, double * kt);
__host__ __device__ int physics_b0(physics_t * phys, double b0[3]);
__host__ __device__ int physics_e0(physics_t * phys, double e0[3]);
__host__ __device__ int physics_e0_frequency(physics_t * phys, double * freq);
__host__ __device__ int physics_e0_flag(physics_t * phys);
__host__ __device__ int physics_fbody(physics_t * phys, double f[3]);
__host__ __device__ int physics_fgrav(physics_t * phys, double g[3]);
__host__ __device__ int physics_mobility(physics_t * phys, double * mobility);
__host__ __device__ int physics_control_timestep(physics_t * phys);
__host__ __device__ int physics_control_time(physics_t * phys, double * t);
__host__ __device__ int physics_fpulse(physics_t * phys, double fpulse[3]);
__host__ __device__ int physics_fpulse_frequency(physics_t * phys, 
						  double * fpule_frequency);

#endif
