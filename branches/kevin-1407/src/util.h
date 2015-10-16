/****************************************************************************
 *
 *  util.h
 *
 *  Utilities with no state to be stored locally.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  (c) 2010-2014 The University of Edinburgh
 *  
 ****************************************************************************/

#ifndef UTIL_H
#define UTIL_H

#include "targetDP.h"

extern const double d_[3][3];
extern const double e_[3][3][3];
extern const double pi_;
extern const double r3_;

extern __targetConst__ double tc_d_[3][3];
extern __targetConst__ double tc_r3_; 

__targetHost__ int    is_bigendian(void);
__targetHost__ double reverse_byte_order_double(char *);
__targetHost__ double dot_product(const double a[3], const double b[3]);
__targetHost__ void   cross_product(const double a[3], const double b[3], double result[3]);
__targetHost__ double modulus(const double a[3]);
__targetHost__ void   rotate_vector(double [3], const double [3]);

__targetHost__ int    imin(const int i, const int j);
__targetHost__ int    imax(const int i, const int j);
__targetHost__ double dmin(const double a, const double b);
__targetHost__ double dmax(const double a, const double b);

__targetHost__ int    util_jacobi(double a[3][3], double vals[3], double vecs[3][3]);
__targetHost__ int    util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]);
__targetHost__ int    util_discrete_volume_sphere(double r0[3], double a0, double * vn);
__targetHost__ int    util_gauss_jordan(const int n, double * a, double * b);
__targetHost__ int    util_dpythag(double a, double b, double * p);
__targetHost__ int    util_svd(int m, int n, double ** u, double * w, double ** v);
__targetHost__ int    util_svd_solve(int m, int n, double ** a, double * b, double * x);
__targetHost__ int    util_matrix_create(int m, int n, double *** p);
__targetHost__ int    util_vector_create(int m, double ** p);
__targetHost__ int    util_matrix_free(int m, double *** p);
__targetHost__ int    util_vector_free(double ** p);
__targetHost__ int util_matrix_invert(int n, double ** a);

__targetHost__ int util_ranlcg_reap_uniform(int * state, double * r);
__targetHost__ int util_ranlcg_reap_gaussian(int * state, double r[2]);

#endif
