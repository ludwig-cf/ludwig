/*****************************************************************************
 *
 *  pe.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *  $Id: pe.h,v 1.2 2008-08-24 17:56:22 kevin Exp $
 *
 *****************************************************************************/

#ifndef _MESSAGES_H
#define _MESSAGES_H

#include <mpi.h>

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

