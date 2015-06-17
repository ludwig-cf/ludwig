/*****************************************************************************
 *
 *  free_energy_tensor.h
 *
 *  Abstract free energy for tensor order parameters.
 *
 *  $Id: free_energy_tensor.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
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

#include "targetDP.h"
#include "free_energy.h"

__targetHost__ void   fe_t_molecular_field_set(void (* f)(const int index, double h[3][3]));
__targetHost__ void   (* fe_t_molecular_field(void))(const int index, double h[3][3]);

__targetHost__ double fe_xi(void);
__targetHost__ void   fe_xi_set(void);

#endif
