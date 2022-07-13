/*****************************************************************************
 *
 *  runtime.h
 *
 *  Runtime input interface.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_RUNTIME_H
#define LUDWIG_RUNTIME_H

#include "pe.h"

typedef enum {RT_NONE, RT_INFO, RT_FATAL} rt_enum_t;

typedef struct rt_s rt_t;

int rt_create(pe_t * pe, rt_t ** prt);
int rt_free(rt_t * rt);
int rt_read_input_file(rt_t * rt, const char * filename);
int rt_info(rt_t * rt);
int rt_int_parameter(rt_t * rt, const char * key, int * ivalue);
int rt_int_parameter_vector(rt_t * rt, const char * key, int ivalue[3]);
int rt_double_parameter(rt_t * rt, const char * key, double * value);
int rt_double_parameter_vector(rt_t * rt, const char * key, double value[3]);
int rt_string_parameter(rt_t * rt, const char * key, char * s,
			unsigned  int len);
int rt_switch(rt_t * rt, const char * key);
int rt_active_keys(rt_t * rt, int * nactive);
int rt_add_key_value(rt_t * rt, const char * key, const char * value);
int rt_int_nvector(rt_t * rt, const char * key, int nv, int * v, rt_enum_t lv);
int rt_double_nvector(rt_t * rt, const char * key, int nv, double * v,
		      rt_enum_t level);
int rt_key_present(rt_t * rt, const char * key);
int rt_key_required(rt_t * rt, const char * key, rt_enum_t level);
int rt_report_unused_keys(rt_t * rt, rt_enum_t level);
int rt_vinfo(rt_t * rt, rt_enum_t lv, const char * fmt, ...);

#endif
