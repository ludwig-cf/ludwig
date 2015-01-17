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

typedef enum {PE_QUIET = 0, PE_VERBOSE} pe_enum_t;

/* At the moment this 'new' interface is coexsiting with ...*/

int pe_create_parent(MPI_Comm parent, pe_t ** pe);
int pe_free(pe_t * pe);
int pe_retain(pe_t * pe);
int pe_commit(pe_t * pe);
int pe_set(pe_t * pe, pe_enum_t option);
int pe_banner(pe_t * pe);
int pe_mpi_comm(pe_t * pe, MPI_Comm * comm);
int pe_mpi_rank(pe_t * pe);
int pe_mpi_size(pe_t * pe);
int pe_subdirectory(pe_t * pe, char * name);
int pe_subdirectory_set(pe_t * pe, const char * name);

/* ... older */
/* please use interface above where possible */

MPI_Comm pe_comm(void);

void info(const char * fmt, ...);
void fatal(const char * fmt, ...);
void verbose(const char * fmt, ...);

#endif
