/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Juho Lintuvuori
 *    Davide Marenduzzo
 *
 *****************************************************************************/

#ifndef BLUEPHASE_H
#define BLUEPHASE_H

/* 'Extension' of free energy (pending free_energy_tensor.h) */

#include "coords.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"

int blue_phase_q_set(field_t * q, field_grad_t * dq);

void   blue_phase_set_free_energy_parameters(double, double, double, double);
void   blue_phase_set_xi(double);
void   blue_phase_set_zeta(double);
void   blue_phase_set_gamma(double);

double blue_phase_free_energy_density(const int);
double blue_phase_compute_fed(double q[3][3], double dq[3][3][3]);
double blue_phase_compute_bulk_fed(double q[3][3]);
double blue_phase_compute_gradient_fed(double q[3][3], double dq[3][3][3]);

void   blue_phase_molecular_field(const int, double h[3][3]);
void   blue_phase_compute_h(double q[3][3], double dq[3][3][3],
			    double dsq[3][3], double h[3][3]);
void   blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
				 double h[3][3], double sth[3][3]);

void   blue_phase_chemical_stress(const int, double sth[3][3]);
void   blue_phase_redshift_set(const double redshift);
void   blue_phase_electric_field_set(const double e[3]);
void   blue_phase_dielectric_anisotropy_set(double e);
double blue_phase_amplitude_compute(void);
double blue_phase_get_xi(void);
double blue_phase_get_zeta(void);
double blue_phase_get_gamma(void);
double blue_phase_chirality(void);
double blue_phase_reduced_temperature(void);
double blue_phase_redshift(void);
double blue_phase_q0(void);
double blue_phase_kappa0(void);
double blue_phase_kappa1(void);
double blue_phase_redshift(void);
double blue_phase_rredshift(void);
double blue_phase_a0(void);
double blue_phase_gamma(void);
double blue_phase_dimensionless_field_strength(void);

void blue_phase_redshift_update_set(int onoff);
int blue_phase_redshift_compute(coords_t * cs);

void blue_phase_q_uniaxial(double amplitude, const double n[3], double q[3][3]);

void blue_phase_set_active_region_gamma_zeta(const int index);

int fed_io_info_set(io_info_t * info);
int  fed_io_info(io_info_t ** info);
int blue_phase_scalar_ops(double q[3][3], double qs[5]);

#endif
 
