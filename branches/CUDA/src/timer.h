/*****************************************************************************
 *
 *  timer.h
 *
 *  Note that the number of timers, their ids, and their descriptive
 *  names must match here.
 *
 *  $Id: timer.h,v 1.4 2010/10/15 12:40:03 kevin Exp $
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

void TIMER_init(void);
void TIMER_start(const int);
void TIMER_stop(const int);
void TIMER_statistics(void);

enum timer_id {TIMER_TOTAL,
	       TIMER_STEPS,
	       TIMER_PROPAGATE,
	       TIMER_COLLIDE,
	       TIMER_HALO_LATTICE,
	       TIMER_PHI_GRADIENTS,
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
	       TIMER_ORDER_PARAMETER_UPDATE,
	       TIMER_FREE1,
	       TIMER_FREE2,
               TIMER_FREE3,
	       PHICOMP,
	       GETPHI,
	       PHIHALO,
	       PHIGRADCOMP,
	       FORCEPUT,
	       PHIPUT,
	       VELOCITYGET,
	       EDGEPACK,
	       EDGEGET,
	       EDGEUNPACK,
	       HALOPACK,
	       HALOPUT,
	       HALOUNPACK};

#endif
