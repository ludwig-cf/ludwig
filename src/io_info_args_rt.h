/*****************************************************************************
 *
 *  io_info_args_rt.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contribuiting authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_INFO_ARGS_RT_H
#define LUDWIG_IO_INFO_ARGS_RT_H

#include "pe.h"
#include "runtime.h"
#include "io_info_args.h"

__host__ int io_info_args_rt(pe_t * pe, rt_t * rt, const char * name,
			     io_info_rw_enum_t rw, io_info_args_t * args);
__host__ int io_info_args_rt_iogrid(pe_t * pe, rt_t * rt, const char * key,
			            int iogrid[3]);

#endif
