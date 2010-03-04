/*****************************************************************************
 *
 *  gelx.h
 *
 *  $Id: gelx.h,v 1.1.2.1 2010-03-04 14:06:46 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef GELX_H
#define GELX_H

double gelx_free_energy_density(const int index);
void   gelx_chemical_stress(const int index, double s[3][3]);

#endif
