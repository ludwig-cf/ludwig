/*****************************************************************************
 *
 *  timer.h
 *
 *  Note that the number of timers, their ids, and their descriptive
 *  names must match here.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_TIMER_H
#define LUDWIG_TIMER_H

#include "pe.h"

/* The aim here is to replace static data in timer.c with something
 * more flexible in timerkeeper_t */

typedef struct timekeeper_options_s timekeeper_options_t;
typedef struct timekeeper_s timekeeper_t;

struct timekeeper_options_s {
  int lap_report;
  int lap_report_freq;
};

struct timekeeper_s {
  pe_t * pe;
  int timestep;
  timekeeper_options_t options;
};

__host__ int timekeeper_create(pe_t * pe, const timekeeper_options_t * opts,
			       timekeeper_t * tk);
__host__ int timekeeper_step(timekeeper_t * tk);
__host__ int timerkeeper_free(timekeeper_t * tk);

__host__ int TIMER_init(pe_t * pe);
__host__ void TIMER_start(const int);
__host__ void TIMER_stop(const int);
__host__ void TIMER_statistics(void);

__host__ double timer_lapse(const int);

enum timer_id {TIMER_TOTAL = 0,
	       TIMER_STEPS,
	       TIMER_PROPAGATE,
	       TIMER_PROP_KERNEL,
	       TIMER_COLLIDE,
	       TIMER_COLLIDE_KERNEL,
	       TIMER_HALO_LATTICE,
	       TIMER_LB_HALO_IMBAL,
	       TIMER_LB_HALO_IRECV,
	       TIMER_LB_HALO_PACK,
	       TIMER_LB_HALO_ISEND,
	       TIMER_LB_HALO_WAIT,
	       TIMER_LB_HALO_UNPACK,
	       TIMER_PHI_GRADIENTS,
	       TIMER_PHI_GRAD_KERNEL,
	       TIMER_PHI_HALO,
	       TIMER_FIELD_HALO_IMBAL,
	       TIMER_FIELD_HALO_IRECV,
	       TIMER_FIELD_HALO_PACK,
	       TIMER_FIELD_HALO_ISEND,
	       TIMER_FIELD_HALO_WAITALL,
	       TIMER_FIELD_HALO_UNPACK,
	       TIMER_LE,
	       TIMER_IO,
	       TIMER_FORCES,
	       TIMER_REBUILD,
	       TIMER_BBL,
	       TIMER_PARTICLE_UPDATE,
	       TIMER_PARTICLE_HALO,
	       TIMER_FLUCTUATIONS,
               TIMER_EWALD_TOTAL,
               TIMER_EWALD_REAL_SPACE,
               TIMER_EWALD_FOURIER_SPACE,
	       TIMER_FORCE_CALCULATION,
	       TIMER_CHEMICAL_STRESS_KERNEL,
	       TIMER_PHI_FORCE_CALC,
	       TIMER_ORDER_PARAMETER_UPDATE,
	       TIMER_U_HALO,
	       TIMER_HYDRO_HALO_IMBAL,
	       TIMER_HYDRO_HALO_IRECV,
	       TIMER_HYDRO_HALO_PACK,
	       TIMER_HYDRO_HALO_ISEND,
	       TIMER_HYDRO_HALO_WAITALL,
	       TIMER_HYDRO_HALO_UNPACK,
	       TIMER_BE_MOL_FIELD,
	       BP_BE_UPDATE_KERNEL, 
	       ADVECTION_X_KERNEL, 
	       ADVECTION_BCS_KERNEL, 
	       ADVECTION_BCS_MEM, 
	       TIMER_ELECTRO_TOTAL,
	       TIMER_ELECTRO_POISSON,
	       TIMER_ELECTRO_NPEQ,
	       TIMER_LAP,
	       TIMER_FREE1,
	       TIMER_FREE2,
               TIMER_FREE3,
	       TIMER_FREE4,
	       TIMER_FREE5,
	       TIMER_FREE6,
	       TIMER_NTIMERS /* This must be the last entry */
};

#endif
