/****************************************************************************
 *
 *  util_ellipsoid.h
 *
 *  Utilities with no state to be stored locally.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  
 ****************************************************************************/

#include <stdint.h>

#ifndef LUDWIG_UTIL_ELLIPSOID_H
#define LUDWIG_UTIL_ELLIPSOID_H

#include "pe.h"
#include "coords.h"

__host__ __device__ void print_vector_onscreen(const double *a, const int n);
__host__ __device__ void print_matrix_onscreen(const double a[3][3]);
__host__ __device__ void normalise_unit_vector(double *a, const int n);
__host__ __device__ void orthonormalise_vector_b_to_a(double *a, double *b);
__host__ __device__ void matrix_product(const double a[3][3], const double b[3][3],
				       double result[3][3]);
__host__ __device__ void matrix_transpose(const double a[3][3], double result[3][3]);
__host__ __device__ void   rotate_byRmatrix(const double R[3][3], const double x[3], double xcap[3]);
__host__ __device__ void   rotationmatrix_from_vectors(const double a[3], const double b[3], double R[3][3]);
__host__ __device__ void   rotationmatrix_from_quaternions(const double q[4], double R[3][3]);
__host__ __device__ void   quaternions_from_dcm(const double R[3][3], double q[4]);
__host__ __device__ void   quaternions_from_vectors(const double a[3], const double b[3], double q[4]);
__host__ __device__ void   eulerangles_from_dcm(const double R[3][3], double *phi, double *theta, double *psi);
__host__ __device__ void   quaternions_from_eulerangles(const double phi, const double theta, const      double psi, double q[4]);
__host__ __device__ void quaternion_product(const double a[4], const double b[4], double result[4]);
__host__ __device__ void rotate_tobodyframe_quaternion(const double q[4], const double a[3], double      b[3]);
__host__ __device__ void rotate_toworldframe_quaternion(const double q[4], const double a[3], double      b[3]);
__host__ __device__ void inertia_tensor_quaternion(const double q[4], const double a[3], double      b[3][3]);
__host__ __device__ void quaternion_from_omega(const double omega[3], const double f, double qbar[4]);
__host__ __device__ void copy_vectortovector(double const a[3], double b[3], const int n);
__host__ __device__ void Jeffery_omega_predicted(double const r, double const quater[4], double const gdot, double opred[3]);
#endif
