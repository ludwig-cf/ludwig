/****************************************************************************
 *
 *  util_ellipsoid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Sumesh Thampi
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_UTIL_ELLIPSOID_H
#define LUDWIG_UTIL_ELLIPSOID_H

#include "pe.h"
#include "coords.h"
void util_q4_product(const double a[4], const double b[4], double c[4]);
int  util_q4_from_euler_angles(double phi, double theta, double psi,
			       double q[4]);
int  util_q4_to_euler_angles(const double q[4], double * phi, double * theta,
			     double * psi);

__host__ __device__ void orthonormalise_vector_b_to_a(double *a, double *b);
__host__ __device__ void matrix_product(const double a[3][3], const double b[3][3], double result[3][3]);
__host__ __device__ void matrix_transpose(const double a[3][3], double result[3][3]);
__host__ __device__ void rotate_tobodyframe_quaternion(const double q[4], const double a[3], double      b[3]);
__host__ __device__ void inertia_tensor_quaternion(const double q[4], const double a[3], double      b[3][3]);
__host__ __device__ void quaternion_from_omega(const double omega[3], const double f, double qbar[4]);
__host__ __device__ void Jeffery_omega_predicted(double const r, double const quater[4], double const gdot, double opred[3], double angpred[2]);
__host__ __device__ void ellipsoid_nearwall_predicted(double const elabc[3], double const h, double const quater[4], double Upred[3], double opred[3]);
 __host__ __device__ void settling_velocity_prolate(double const r, double const f, double const mu, double const ela, double U[2]);
__host__ __device__ void euler_from_vectors(double a[3], double b[3], double *euler);
 __host__ __device__ void euler_from_dcm(double const r[3][3], double *phi, double *theta, double *psi);
__host__ __device__ void dcm_from_vectors(double const a[3], double const b[3], double const c[3], double r[3][3]);
__host__ __device__ double mass_ellipsoid(const double dim[3], const double density);
__host__ __device__ void unsteady_mI(const double q[4], const double I[3], const double omega[3], double F[3][3]);

#endif
