/*****************************************************************************
 *
 *  advection.h
 *
 *  $Id: advection.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_H
#define ADVECTION_H

void advection_order_set(const int order);
int  advection_order(void);
void advection_order_n(double * fluxe, double * fluxw, double * fluxy,
		       double * fluxz);
void advection_upwind(double * fluxe, double * fluxw, double * fluxy,
		      double * fluxz);
void advection_second_order(double * fluxe, double * fluxw, double * fluxy,
			    double * fluxz);
void advection_fourth_order(double * fluxe, double * fluxw, double * fluxy,
			    double * fluxz);
void advection_upwind_third_order(double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz);
void advection_upwind_fifth_order(double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz);
void advection_upwind_seventh_order(double * fluxe, double * fluxw,
				    double * fluxy, double * fluxz);

#endif
