/****************************************************************************
 *
 *  util_ellipsoid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Sumesh Thampi introduced ellipsoids.
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_UTIL_ELLIPSOID_H
#define LUDWIG_UTIL_ELLIPSOID_H

#include "pe.h"
#include "coords.h"

void util_q4_product(const double a[4], const double b[4], double c[4]);
void util_q4_rotate_vector(const double q[4], const double a[3], double b[3]);
void util_q4_from_omega(const double omega[3], double dt, double q[4]);
int  util_q4_from_euler_angles(double phi, double theta, double psi,
			       double q[4]);
int  util_q4_to_euler_angles(const double q[4], double * phi, double * theta,
			     double * psi);
int  util_q4_is_inside_ellipsoid(const double q[4], const double elabc[3],
				 const double r[3]);
void util_q4_inertia_tensor(const double q[4], const double moment[3],
			    double mI[3][3]);

int util_ellipsoid_is_sphere(const double elabc[3]);
int util_spheroid_surface_normal(const double elabc[3], const double m[3],
				 const double r[3], double v[3]);
int util_spheroid_surface_tangent(const double elabc[3], const double m[3],
				  const double r[3], double vt[3]);
int util_ellipsoid_euler_from_vectors(const double a0[3], const double b0[3],
				      double euler[3]);
int util_ellipsoid_prolate_settling_velocity(double a, double b, double eta,
					     double f, double u[2]);

void matrix_product(const double a[3][3], const double b[3][3],
		    double result[3][3]);
void matrix_transpose(const double a[3][3], double result[3][3]);

#endif
