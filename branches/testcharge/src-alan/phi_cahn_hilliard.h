/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: phi_cahn_hilliard.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_CAHN_HILLIARD_H
#define PHI_CAHN_HILLIARD_H


#define HOST
#ifdef INCLUDED_FROM_TARGET
#ifdef CUDA
#define HOST extern "C"
#endif
#endif

void   phi_cahn_hilliard(void);
void   phi_cahn_hilliard_mobility_set(const double);
HOST double phi_cahn_hilliard_mobility(void);

#endif
