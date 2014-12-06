/*****************************************************************************
 *
 *  ran.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef RAN_H
#define RAN_H

#include "runtime.h"

int  ran_init(rt_t * rt);
double ran_parallel_gaussian(void);
double ran_parallel_uniform(void);
void   ran_parallel_unit_vector(double []);
double ran_serial_uniform(void);
double ran_serial_gaussian(void);
void   ran_serial_unit_vector(double []);

#endif
