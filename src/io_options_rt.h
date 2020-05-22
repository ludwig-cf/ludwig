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
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_OPTIONS_RT_H
#define LUDWIG_IO_OPTIONS_RT_H

#include "pe.h"
#include "runtime.h"
#include "io_options.h"

__host__ int io_options_rt(pe_t * pe, rt_t * rt, const char * keystub,
			   io_options_t * opts);
__host__ int io_options_rt_mode(pe_t * pe, rt_t * rt, const char * key,
				io_mode_enum_t * mode);
__host__ int io_options_rt_rformat(pe_t * pe, rt_t * rt, const char * key,
				   io_rformat_enum_t * options);
#endif
