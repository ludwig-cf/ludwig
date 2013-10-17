#ifdef PETSC
/*****************************************************************************
 *
 *  psi_petsc.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2013 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PSI_PETSC_H
#define PSI_PETSC_H

#include "psi.h"

int psi_petsc_init(psi_t * obj);
int psi_petsc_solve(psi_t * obj, f_vare_t fepsilon);
int psi_petsc_poisson(psi_t * obj);

#endif
#endif

