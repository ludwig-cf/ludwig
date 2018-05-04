/*****************************************************************************
 *
 *  ran.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef RAN_H
#define RAN_H

#include "pe.h"
#include "runtime.h"

/* This static generator is scheduled for removal. Prefer noise_t
 * or routines in util.h */

int ran_init(pe_t * pe);
int ran_init_rt(pe_t * pe, rt_t * rt);
int ran_init_seed(pe_t * pe, int scalar_seed);

double ran_parallel_gaussian(void);
double ran_parallel_uniform(void);
double ran_serial_uniform(void);
double ran_serial_gaussian(void);

#endif
