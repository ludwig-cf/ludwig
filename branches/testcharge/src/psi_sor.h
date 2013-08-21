/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2013 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#ifndef PSI_SOR_H
#define PSI_SOR_H

/* f_vare_t describes the signature of the function expected
 * to return the permeativity as a function of position index. */

typedef int (* f_vare_t)(int index, double * epsilon);

#include "psi.h"

int psi_sor_poisson(psi_t * obj);
int psi_sor_vare_poisson(psi_t * obj, f_vare_t fepsilon);
int psi_sor_solve(psi_t * obj, f_vare_t fepsilon);
int psi_sor_offset(psi_t * obj);

#endif
