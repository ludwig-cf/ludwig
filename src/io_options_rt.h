/*****************************************************************************
 *
 *  io_options_rt.h
 *
 *  Runtime parsing of i/o options.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_OPTIONS_RT_H
#define LUDWIG_IO_OPTIONS_RT_H

#include "runtime.h"
#include "io_options.h"

__host__ int io_options_rt(rt_t * rt, rt_enum_t lv, const char * keystub,
			   io_options_t * opts);
__host__ int io_options_rt_mode(rt_t * rt, rt_enum_t lv, const char * key,
				io_mode_enum_t * mode);
__host__ int io_options_rt_record_format(rt_t * rt, rt_enum_t lv,
					 const char * key,
					 io_record_format_enum_t * options);
__host__ int io_options_rt_report(rt_t * rt, rt_enum_t lv, const char * key,
				  int * report);
#endif
