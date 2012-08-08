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

#include "hydro.h"

#ifdef OLD_PHI
int    phi_cahn_hilliard(hydro_t * hydro);
#else
#include "field.h"
int phi_cahn_hilliard(field_t * phi, hydro_t * hydro);
#endif
void   phi_cahn_hilliard_mobility_set(const double);
double phi_cahn_hilliard_mobility(void);

#endif
