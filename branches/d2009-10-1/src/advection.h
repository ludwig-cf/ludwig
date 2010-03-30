/*****************************************************************************
 *
 *  advection.h
 *
 *  $Id: advection.h,v 1.1.4.1 2010-03-30 14:12:26 kevin Exp $
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

void advection_order_set(const int order);
int  advection_order(void);
void advection_order_n(double * fluxw, double * fluxe, double * fluxy,
		       double * fluxz);
void advection_upwind(double * fluxw, double * fluxe, double * fluxy,
		      double * fluxz);
void advection_upwind_third_order(double * fluxw, double * fluxe,
				  double * fluxy, double * fluxz);
void advection_upwind_fifth_order(double * fluxw, double * fluxe,
				  double * fluxy, double * fluxz);
void advection_upwind_seventh_order(double * fluxw, double * fluxe,
				    double * fluxy, double * fluxz);

#endif
