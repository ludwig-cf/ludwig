/*****************************************************************************
 *
 *  io_info_args_rt.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2022 The University of Edinburgh
 *
 *  Contribuiting authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_INFO_ARGS_RT_H
#define LUDWIG_IO_INFO_ARGS_RT_H

#include "runtime.h"
#include "io_info_args.h"

__host__ int io_info_args_rt(rt_t * rt, rt_enum_t lv, const char * name,
			     io_info_rw_enum_t rw, io_info_args_t * args);
__host__ int io_info_args_rt_input(rt_t * rt, rt_enum_t lv, const char * stub,
				   io_info_args_t * args);
__host__ int io_info_args_rt_output(rt_t * rt, rt_enum_t lv, const char * stub,
				    io_info_args_t * args);
__host__ int io_info_args_rt_iogrid(rt_t * rt, rt_enum_t lv, const char * key,
			            int iogrid[3]);

#endif
