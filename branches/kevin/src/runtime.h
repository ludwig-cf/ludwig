/*****************************************************************************
 *
 *  runtime.h
 *
 *  Runtime input interface.
 *
 *  $Id: runtime.h,v 1.3 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef RUNTIME_H
#define RUNTIME_H

#include "pe.h"

typedef struct rt_s rt_t;

int rt_create(pe_t * pe, rt_t ** prt);
int rt_free(rt_t ** rt);
int rt_read_input_file(rt_t * rt, const char * filename);
int rt_int_parameter(rt_t * rt, const char * key, int * ivalue);
int rt_int_parameter_vector(rt_t * rt, const char * key, int ivalue[3]);
int rt_double_parameter(rt_t * rt, const char * key, double * value);
int rt_double_parameter_vector(rt_t * rt, const char * key, double value[3]);
int rt_string_parameter(rt_t * rt, const char * key, char * s, const int len);
int rt_switch(rt_t * rt, const char * key);
int rt_active_keys(rt_t * rt, int * nactive);

#endif
