
#ifndef _RAN_H
#define _RAN_H

/*****************************************************************************
 *
 *  ran.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

void   ran_init(void);
double ran_parallel_gaussian(void);
double ran_parallel_uniform(void);
double ran_serial_uniform(void);
double ran_serial_gaussian(void);

#endif /* _RAN_H */
