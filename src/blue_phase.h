/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  $Id: blue_phase.h,v 1.5 2010-10-15 12:40:02 kevin Exp $
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

void   blue_phase_set_free_energy_parameters(double, double, double, double);
void   blue_phase_set_xi(double);

double blue_phase_free_energy_density(const int);
double blue_phase_compute_fed(double q[3][3], double dq[3][3][3]);

void   blue_phase_molecular_field(const int, double h[3][3]);
void   blue_phase_compute_h(double q[3][3], double dq[3][3][3],
			    double dsq[3][3], double h[3][3]);
void   blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
				 double h[3][3], double sth[3][3]);

void   blue_phase_chemical_stress(const int, double sth[3][3]);
void   blue_phase_O8M_init(double amplitude);
void   blue_phase_O2_init(double amplitude);
void   blue_phase_redshift_set(const double redshift);

double blue_phase_get_xi(void);
double blue_phase_chirality(void);
double blue_phase_reduced_temperature(void);
double blue_phase_redshift(void);

void blue_phase_twist_init(double amplitude);
void blue_set_random_q_init(double xmin, double xmax, double ymin, double ymax,
			    double zmin, double zmax);
#endif
 
