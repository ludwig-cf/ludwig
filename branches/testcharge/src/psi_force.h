/*****************************************************************************
 *
 *  psi_force.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013 The University of Edinburgh
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    
 *****************************************************************************/

#ifndef PSI_FORCE_H
#define PSI_FORCE_H

#include "psi.h"
#include "hydro.h"

int psi_force_grad_mu(psi_t * psi, hydro_t * hydro, double dt);
int psi_force_external_field(psi_t * psi, hydro_t * hydro, double dt);
int psi_force_gradmu_conserve(psi_t * psi,  hydro_t * hydro, double dt);

#endif
