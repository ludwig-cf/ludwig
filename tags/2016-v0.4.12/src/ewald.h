/*****************************************************************************
 *
 *  ewald.h
 *
 *  $Id: ewald.h,v 1.3 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *    Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Grace Kim
 *
 *****************************************************************************/

#ifndef EWALD_H
#define EWALD_H

typedef struct ewald_s ewald_t;

int ewald_create(double mu, double rc, colloids_info_t * cinfo, ewald_t ** e);
void ewald_free(ewald_t * ewald);
int ewald_info(ewald_t * ewald);
int ewald_kappa(ewald_t * ewald, double * kappa);
int ewald_sum(ewald_t * ewald);
int ewald_real_space_sum(ewald_t * ewald);
int ewald_fourier_space_sum(ewald_t * ewald);

int ewald_total_energy(ewald_t * ewald, double * ereal, double * efourier,
		       double * eself);
int ewald_fourier_space_energy(ewald_t * ewald, double * ef);
int ewald_self_energy(ewald_t * ewald, double * es);
int ewald_real_space_energy(ewald_t * ewald, const double r1[3],
			    const double r2[3], const double r12[3],
			    double * er);
#endif
