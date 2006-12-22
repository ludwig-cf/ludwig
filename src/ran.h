
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
void   ran_parallel_unit_vector(double []);
double ran_serial_uniform(void);
double ran_serial_gaussian(void);
void   ran_serial_unit_vector(double []);

#endif /* _RAN_H */
