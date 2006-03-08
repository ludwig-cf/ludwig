/*****************************************************************************
 *
 *  runtime.h
 *
 *  Runtime input interface.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _RUNTIME_H
#define _RUNTIME_H

void RUN_read_input_file(const char *);
int  RUN_get_int_parameter(const char *, int *);
int  RUN_get_int_parameter_vector(const char *, int []);
int  RUN_get_double_parameter(const char *, double *);
int  RUN_get_double_parameter_vector(const char *, double []);
int  RUN_get_string_parameter(const char *, char *);
int  RUN_get_active_keys(void);

#endif
