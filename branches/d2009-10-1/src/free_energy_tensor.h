/*****************************************************************************
 *
 *  free_energy_tensor.h
 *
 *  Abstract free energy for tensor order parameters.
 *
 *  $Id: free_energy_tensor.h,v 1.1.2.1 2010-03-21 13:43:23 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 the University of Edinburgh
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_TENSOR_H
#define FREE_ENERGY_TENSOR_H

/* We 'extend' the free energy abstract type via... */
#include "free_energy.h"

void   fe_t_molecular_field_set(void (* f)(const int index, double h[3][3]));
void   (* fe_t_molecular_field(void))(const int index, double h[3][3]);

double fe_xi(void);
void   fe_xi_set(void);

#endif
