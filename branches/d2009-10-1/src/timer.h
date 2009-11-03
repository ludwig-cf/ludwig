/*****************************************************************************
 *
 *  timer.h
 *
 *  Note that the number of timers, their ids, and their descriptive
 *  names must match here.
 *
 *  The macros are provided to allow timing calls to be inserted into
 *  performance sensitive regions of code.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _TIMER_H
#define _TIMER_H

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
	       TIMER_FREE1,
	       TIMER_FREE2,
               TIMER_FREE3};

#ifdef _TIMER_MACRO_ON_
  #define TIMER_START_MACRO(A) TIMER_start A
  #define TIMER_STOP_MACRO(A)  TIMER_stop A
#else
  #define TIMER_START_MACRO(A)
  #define TIMER_STOP_MACRO(A)
#endif

#endif

