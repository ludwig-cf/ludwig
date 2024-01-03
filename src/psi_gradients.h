/****************************************************************************
 *
 *  psi_gradients.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  (c) 2014-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk) (now U. Strathclyde)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef  LUDWIG_PSI_GRADIENTS_H
#define  LUDWIG_PSI_GRADIENTS_H

#include "psi.h"

int psi_electric_field(psi_t * psi, int index, double e[3]);

#endif                     
