/*****************************************************************************
 *
 *  pe.h
 *
 *  $Id: pe.h,v 1.3 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PE_H
#define PE_H

#include <mpi.h>

void pe_init(int , char **);
void pe_finalise(void);
int  pe_rank(void);
int  pe_size(void);
void pe_parent_comm_set(const MPI_Comm parent);

MPI_Comm pe_comm(void);

void info(const char * fmt, ...);
void fatal(const char * fmt, ...);
void verbose(const char * fmt, ...);

#endif
