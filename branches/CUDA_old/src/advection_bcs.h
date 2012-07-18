/*****************************************************************************
 *
 *  advection_bcs.h
 *
 *  $Id: advection_bcs.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef ADVECTION_BCS_H
#define ADVECTION_BCS_H

void advection_bcs_no_normal_flux(double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz);
void advection_bcs_wall(void);

#endif
