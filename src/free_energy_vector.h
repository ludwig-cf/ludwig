/*****************************************************************************
 *
 *  free_energy_vector.h
 *
 *  Abstract free energy for vector order parameters.
 *
 *  $Id: free_energy_vector.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 the University of Edinburgh
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_VECTOR_H
#define FREE_ENERGY_VECTOR_H

#include "targetDP.h"
#include "free_energy.h"

HOST void   fe_v_molecular_field_set(void (* f)(const int index, double h[3]));
HOST void   (* fe_v_molecular_field(void))(const int index, double h[3]);

HOST double fe_v_lambda(void);
HOST void   fe_v_lambda_set(const double);

#endif
