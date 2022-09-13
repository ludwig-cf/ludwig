/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_SOR_H
#define LUDWIG_PSI_SOR_H

#include "psi.h"
#include "fe_electro_symmetric.h"

int psi_sor_solve(psi_t * obj, fe_t * fe, f_vare_t fepsilon, int its);
int psi_sor_poisson(psi_t * obj, int its);
int psi_sor_vare_poisson(psi_t * obj, fe_es_t * fe, f_vare_t fepsilon, int its);

#endif
