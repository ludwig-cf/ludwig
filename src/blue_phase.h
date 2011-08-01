/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUEPHASE_H
#define BLUEPHASE_H

/* 'Extension' of free energy (pending free_energy_tensor.h) */

#include "free_energy.h"

void   blue_phase_set_free_energy_parameters(double, double, double, double);
void   blue_phase_set_xi(double);
void   blue_phase_set_zeta(double);
void   blue_phase_set_gamma(double);

double blue_phase_free_energy_density(const int);
double blue_phase_compute_fed(double q[3][3], double dq[3][3][3]);

void   blue_phase_molecular_field(const int, double h[3][3]);
void   blue_phase_compute_h(double q[3][3], double dq[3][3][3],
			    double dsq[3][3], double h[3][3]);
void   blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
				 double h[3][3], double sth[3][3]);

void   blue_phase_chemical_stress(const int, double sth[3][3]);
void   blue_phase_O8M_init(void);
void   blue_phase_O2_init(void);
void   blue_phase_O5_init(void);
void   blue_phase_H2D_init(void);
void   blue_phase_H3DA_init(void);
void   blue_phase_H3DB_init(void);
void   blue_phase_DTC_init(void);
void   blue_phase_redshift_set(const double redshift);
void   blue_phase_amplitude_set(const double amplitude);
void   blue_phase_electric_field_set(const double e[3]);
void   blue_phase_dielectric_anisotropy_set(double e);

double blue_phase_amplitude(void);
double blue_phase_get_xi(void);
double blue_phase_get_zeta(void);
double blue_phase_get_gamma(void);
double blue_phase_chirality(void);
double blue_phase_reduced_temperature(void);
double blue_phase_redshift(void);
double blue_phase_q0(void);
double blue_phase_dimensionless_field_strength(void);

void blue_phase_redshift_update_set(int onoff);
void blue_phase_redshift_compute(void);
void blue_phase_stats(int nstep);

void blue_phase_q_uniaxial(const double n[3], double q[3][3]);
void blue_phase_nematic_init(const double n[3]);
void blue_phase_twist_init(const int helical_axis);
void blue_phase_chi_edge(int N, double z0, double x0);
void blue_set_random_q_init(void);
void blue_set_random_q_rectangle_init(const double xmin, const double xmax,
				      const double ymin,const double ymax,
				      const double zmin,const double zmax);

void blue_phase_set_active_region_gamma_zeta(const int index);
#endif
 
