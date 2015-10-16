/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2013 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#ifndef PSI_SOR_H
#define PSI_SOR_H

#include "psi.h"

int psi_sor_poisson(psi_t * obj);
int psi_sor_vare_poisson(psi_t * obj, f_vare_t fepsilon);
int psi_sor_solve(psi_t * obj, f_vare_t fepsilon);

#endif
