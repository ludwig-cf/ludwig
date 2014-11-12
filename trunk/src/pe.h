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

HOST void pe_init(void);
HOST void pe_finalise(void);
HOST int  pe_init_quiet(void);
HOST int  pe_rank(void);
HOST int  pe_size(void);
HOST void pe_parent_comm_set(const MPI_Comm parent);
HOST void pe_redirect_stdout(const char * filename);
HOST void pe_subdirectory_set(const char * name);
HOST void pe_subdirectory(char * name);

HOST MPI_Comm pe_comm(void);

HOST void info(const char * fmt, ...);
HOST void fatal(const char * fmt, ...);
HOST void verbose(const char * fmt, ...);

#endif
