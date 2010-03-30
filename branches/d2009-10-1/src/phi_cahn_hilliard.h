/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: phi_cahn_hilliard.h,v 1.4.4.1 2010-03-30 14:12:26 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_CAHN_HILLIARD_H
#define PHI_CAHN_HILLIARD_H

void phi_cahn_hilliard(void);
void phi_ch_set_langmuir_hinshelwood(double kplus, double kminus, double psi);

double phi_ch_get_mobility(void);
void   phi_ch_set_mobility(const double);
void   phi_ch_op_set_mobility(const double, const int);

#endif
