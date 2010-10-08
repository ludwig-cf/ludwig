/*****************************************************************************
 *
 *  bbl.h
 *
 *  $Id: bbl.h,v 1.2.16.1 2010-10-08 15:29:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BBL_H
#define BBL_H

void bounce_back_on_links(void);
void bbl_surface_stress(void);
void bbl_active_on_set(void);
int  bbl_active_on(void);
double bbl_order_parameter_deficit(void);

#endif
