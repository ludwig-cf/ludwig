/*****************************************************************************
 *
 *  io_aggr_mpio.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_IO_AGGR_MPIO_H
#define LUDWIG_IO_AGGR_MPIO_H

#include "pe.h"
#include "coords.h"
#include "io_aggregator.h"

int io_aggr_mpio_write(pe_t * pe, cs_t * cs, const char * filename,
		       const io_aggregator_t * buf);
int io_aggr_mpio_read(pe_t * pe, cs_t * cs, const char * filename,
		      io_aggregator_t * buf);
#endif
