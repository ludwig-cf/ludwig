/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  $Id: blue_phase.h,v 1.2 2009-06-01 16:50:07 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef BLUEPHASE_H
#define BLUEPHASE_H

double blue_phase_free_energy_density(const int);
void   blue_phase_molecular_field(const int, double h[3][3]);
void   blue_phase_chemical_stress(const int, double sth[3][3]);
void   blue_phase_O8M_init(void);

double blue_phase_chirality(void);
double blue_phase_reduced_temperature(void);
#endif
 
