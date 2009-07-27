/*****************************************************************************
 *
 *  advection.h
 *
 *  $Id: advection.h,v 1.1 2009-07-27 13:48:34 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#ifndef ADVECTION_H
#define ADVECTION_H

void advection_upwind(double *, double *, double *, double *);
void advection_upwind_third_order(double *, double *, double *, double *);
void advection_upwind_fifth_order(double *, double *, double *, double *);
void advection_upwind_seventh_order(double *, double *, double *, double *);

#endif
