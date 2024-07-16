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
int util_ellipsoid_euler_from_vectors(const double a0[3], const double b0[3],
				      double euler[3]);
int util_ellipsoid_prolate_settling_velocity(double a, double b, double eta,
					     double f, double u[2]);
int util_discrete_volume_ellipsoid(const double abc[3], const double r0[3],
				   const double q[4], double * vol);

void matrix_product(const double a[3][3], const double b[3][3],
		    double result[3][3]);
void matrix_transpose(const double a[3][3], double result[3][3]);

/*****************************************************************************
 *
 *  __host__ __device__ static inline
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "target.h"
#include "util_vector.h"

/*****************************************************************************
 *
 *  util_ellipsoid_is_sphere
 *
 *  Return 1 if a = b = c.
 *
 *****************************************************************************/

__host__ __device__
static inline int util_ellipsoid_is_sphere(const double elabc[3]) {

  double a = elabc[0];
  double b = elabc[1];
  double c = elabc[2];

  return ((fabs(b - a) < DBL_EPSILON) && (fabs(c - b) < DBL_EPSILON));
}

/*****************************************************************************
 *
 *  util_spheroid_normal_tangent
 *
 *  Calculate surface normal or tangent on a spheroid at a point r
 *  which may be at, or near, the surface.
 *
 *  The computation is rather similar: tangent = 1 gives tangent,
 *  otherwise the normal is returned.
 *
 *  Note
 *  "Spheroid" has a != b and b = c (for oblate or prolate case).
 *
 *  See, e.g., S.R. Keller and T.Y.T. Wu, J. Fluid Mech. 80, 259--278 (1977).
 *
 *  CHECK what happens in the oblate case?
 *
 *  The vector returned should be a unit vector if r is exactly at the
 *  surface, but may not be in other cases.
 *
 ****************************************************************************/

__host__ __device__
static inline int util_spheroid_normal_tangent(const double elabc[3],
					       const double elbz[3],
					       const double r[3],
					       int tangent,
					       double result[3]) {
  int ifail = 0;
  double elc;
  double ele,ele2;
  double ela,ela2;
  double elz,elz2;
  double elr;
  double rmod;
  double elrho[3];
  double diff1, diff2;

  /* elabc[0] must be a, the largest dimension, and b == c */
  assert(elabc[0] >  elabc[1]);
  assert(fabs(elabc[1] - elabc[2]) < DBL_EPSILON);

  ela = elabc[0];
  elc = sqrt(elabc[0]*elabc[0] - elabc[1]*elabc[1]);
  ele = elc/ela;
  elz = util_vector_dot_product(r, elbz);

  for (int ia = 0; ia < 3; ia++) {
    elrho[ia] = r[ia] - elz*elbz[ia];
  }
  elr = util_vector_modulus(elrho);
  rmod = 0.0;
  if (elr != 0.0) rmod = 1.0/elr;

  for (int ia = 0; ia < 3; ia++) {
    elrho[ia] = elrho[ia]*rmod;
  }

  ela2  = ela*ela;
  elz2  = elz*elz;
  ele2  = ele*ele;
  diff1 = ela2 - elz2;
  diff2 = ela2 - ele2*elz2;

  /* If r is not exactly at the surface, then elz > ela. An adjustment
   * is made ... */

  if (diff1 < 0.0) {
    double dr[3] = {0};
    double gridin[3] = {0};
    double elzin = 0.0;
    elr = util_vector_modulus(r);
    rmod = 0.0;
    if (elr != 0.0) rmod = 1.0/elr;
    for (int ia = 0; ia < 3; ia++) {
      dr[ia] = r[ia]*rmod;
    }
    for (int ia = 0; ia < 3; ia++) {
      gridin[ia] = r[ia] - dr[ia];
    }
    elzin = util_vector_dot_product(gridin, elbz);
    elz2  = elzin*elzin;
    diff1 = ela2 - elz2;

    if (diff2 < 0.0) diff2 = ela2 - ele2*elz2;
  }

  assert(diff1 >= 0.0);
  assert(diff2 >  0.0);

  {
    double denom = sqrt(diff2);
    double term1 = sqrt(diff1)/denom;
    double term2 = sqrt(1.0 - ele*ele)*elz/denom;

    if (tangent) {
      /* Tangent vector */
      for (int ia = 0; ia < 3; ia++) {
	result[ia] = -term1*elbz[ia] + term2*elrho[ia];
      }
      elr = util_vector_modulus(elrho);
      if (elr <= 0.0) ifail = -999; /* tangent fails */
    }
    else {
      /* Normal vector */
      for (int ia = 0; ia < 3; ia++) {
	result[ia] = +term2*elbz[ia] + term1*elrho[ia];
      }
    }
    /* If r is exactly at the surface, the result will be a unit vector. */
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_spheroid_surface_normal
 *
 *  Calculate surface normal on an spheroid.
 *  elabc[3]  - a >= b = c for spheroid here
 *  m[3]      - vector aligned along principal axis a
 *  r[3]      - position at surface
 *  v[3]      - normal vector at position r
 *
 ****************************************************************************/

__host__ __device__
static inline int util_spheroid_surface_normal(const double elabc[3],
					       const double m[3],
					       const double r[3],
					       double v[3]) {

  return util_spheroid_normal_tangent(elabc, m, r, 0, v);
}

/*****************************************************************************
 *
 *  util_spheroid_surface_tangent
 *
 *  Calculate surface tangent on a spheroid.
 *  See comments above for normal version, except the result is the tangent.
 *
 *****************************************************************************/

__host__ __device__
static inline int util_spheroid_surface_tangent(const double elabc[3],
						const double m[3],
						const double r[3],
						double vt[3]) {

  return util_spheroid_normal_tangent(elabc, m, r, 1, vt);
}

#endif
