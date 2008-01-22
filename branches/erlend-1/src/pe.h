/*****************************************************************************
 *
 *  pe.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _MESSAGES_H
#define _MESSAGES_H


#ifdef _MPI_
#include <mpi.h>
#endif


void pe_init(int , char **);
void pe_finalise(void);
int  pe_rank(void);
int  pe_size(void);

void info(const char *, ...);
void fatal(const char *, ...);
void verbose(const char *, ...);

int imin(const int, const int);
int imax(const int, const int);
double dmin(const double, const double);
double dmax(const double, const double);


#ifdef _VERBOSE_MACRO_ON_
#define VERBOSE(A) verbose A
#else
#define VERBOSE(A)
#endif

#endif /* _MESSAGES_H */

