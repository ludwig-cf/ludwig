/*****************************************************************************
 *
 *  ewald.h
 *
 *  $Id: ewald.h,v 1.1.2.1 2007-04-30 14:37:22 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef EWALD_H_
#define EWALD_H_

void ewald_init(double);
void ewald_sum(void);
double ewald_real_space_energy(double [], double [], double []);
double ewald_fourier_space_energy(void);
double ewald_self_energy(void);

void   ewald_test(void);
void   ewald_total_energy(double *, double *, double *);
#endif
