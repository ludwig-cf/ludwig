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
 *  $Id: phi_cahn_hilliard.h,v 1.4 2009-07-27 13:48:34 kevin Exp $
 *
 *****************************************************************************/

#ifndef PHICAHNHILLIARD_H
#define PHICAHNHILLIARD_H

void phi_cahn_hilliard(void);
void phi_ch_set_upwind_order(int);
void phi_ch_set_langmuir_hinshelwood(double kplus, double kminus, double psi);

double phi_ch_get_mobility(void);
void   phi_ch_set_mobility(const double);
void   phi_ch_op_set_mobility(const double, const int);

#endif
