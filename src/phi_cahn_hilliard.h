/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh 2008
 *
 *  $Id: phi_cahn_hilliard.h,v 1.2 2008-08-24 16:58:10 kevin Exp $
 *
 *****************************************************************************/

#ifndef _PHICAHNHILLIARD
#define _PHICAHNHILLIARD

void phi_cahn_hilliard(void);
void phi_ch_set_upwind_order(int);

double phi_ch_get_mobility(void);
void   phi_ch_set_mobility(const double);

#endif
