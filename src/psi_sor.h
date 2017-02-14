/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PSI_SOR_H
#define PSI_SOR_H

#include "psi.h"
#include "fe_electro_symmetric.h"

int psi_sor_solve(psi_t * obj, fe_t * fe, f_vare_t fepsilon);
int psi_sor_poisson(psi_t * obj);
int psi_sor_vare_poisson(psi_t * obj, fe_es_t * fe, f_vare_t fepsilon);

#endif
