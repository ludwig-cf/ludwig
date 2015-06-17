/*****************************************************************************
 *
 *  pe.h
 *
 *  (c) 2010-2014 The University of Edinburgh
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PE_H
#define PE_H

#include "../version.h"
#include <mpi.h>
#include "targetDP.h"

typedef struct pe_s pe_t;

__targetHost__ void pe_init(void);
__targetHost__ void pe_finalise(void);
__targetHost__ int  pe_init_quiet(void);
__targetHost__ int  pe_rank(void);
__targetHost__ int  pe_size(void);
__targetHost__ void pe_parent_comm_set(const MPI_Comm parent);
__targetHost__ void pe_redirect_stdout(const char * filename);
__targetHost__ void pe_subdirectory_set(const char * name);
__targetHost__ void pe_subdirectory(char * name);

__targetHost__ MPI_Comm pe_comm(void);

__targetHost__ void info(const char * fmt, ...);
__targetHost__ void fatal(const char * fmt, ...);
__targetHost__ void verbose(const char * fmt, ...);

#endif
