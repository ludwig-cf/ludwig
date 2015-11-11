/*****************************************************************************
 *
 *  timer.h
 *
 *  Note that the number of timers, their ids, and their descriptive
 *  names must match here.
 *
 *  $Id: timer.h,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef TIMER_H
#define TIMER_H

__targetHost__ void TIMER_init(void);
__targetHost__ void TIMER_start(const int);
__targetHost__ void TIMER_stop(const int);
__targetHost__ void TIMER_statistics(void);

enum timer_id {TIMER_TOTAL = 0,
	       TIMER_STEPS,
	       TIMER_PROPAGATE,
	       TIMER_COLLIDE,
	       TIMER_COLLIDE_KERNEL,
	       TIMER_HALO_LATTICE,
	       TIMER_PHI_GRADIENTS,
	       TIMER_PHI_HALO,
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
	       TIMER_ORDER_PARAMETER_UPDATE,
	       TIMER_U_HALO, 
	       BP_BE_UPDATE_KERNEL, 
	       ADVECTION_X_KERNEL, 
	       ADVECTION_BCS_KERNEL, 
	       ADVECTION_BCS_MEM, 
	       TIMER_ELECTRO_TOTAL,
	       TIMER_ELECTRO_POISSON,
	       TIMER_ELECTRO_NPEQ,
	       TIMER_FREE1,
	       TIMER_FREE2,
               TIMER_FREE3,
	       TIMER_NTIMERS /* This must be the last entry */
};

#endif
