/*****************************************************************************
 *
 *  fluctuations.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FLUCTUATIONS_H
#define FLUCTUATIONS_H

#define NFLUCTUATION 10
#define NFLUCTUATION_STATE 4

typedef struct fluctuations_type fluctuations_t;

fluctuations_t * fluctuations_create(const int nsites);
unsigned int     fluctuations_uniform(unsigned int state[NFLUCTUATION_STATE]);
void             fluctuations_state_set(fluctuations_t * f_t, const int index,
		              const unsigned int state[NFLUCTUATION_STATE]);
void             fluctuations_state(const fluctuations_t * f_t,
				    const int index,
				    unsigned int state[NFLUCTUATION_STATE]);
void             fluctuations_reap(fluctuations_t * f_t, const int index,
				   double * reap);
void             fluctuations_destroy(fluctuations_t * f_t);

#endif
