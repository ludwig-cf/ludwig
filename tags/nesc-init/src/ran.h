/*****************************************************************************
 *
 *  ran.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _RAN_H
#define _RAN_H

void   RAN_init(void);
double RAN_uniform(void);

double ran_parallel_gaussian(void);
double ran_parallel_uniform(void);
double ran_serial_uniform(void);
double ran_serial_gaussian(void);

#endif /* _RAN_H */
