/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2012)
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PSI_SOR_H
#define PSI_SOR_H

/* f_var_t describes the signature of the function expected
 * to return the permeativity as a function of position index. */

typedef int (* f_vare_t)(int index, double * epsilon);

#include "psi.h"

int psi_sor_poisson(psi_t * obj);
int psi_sor_vare_poisson(psi_t * obj, f_vare_t fepsilon);

#endif
