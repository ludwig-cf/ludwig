/****************************************************************************
 *
 *  util.h
 *
 *  Utilities with no state to be stored locally.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  
 ****************************************************************************/

#include <stdint.h>

#ifndef LUDWIG_UTIL_H
#define LUDWIG_UTIL_H

/* These are needed for a number of declarations [X,Y,Z] etc
 * but there are no functional dependencies. */

#include "pe.h"
#include "coords.h"

#define KRONECKER_DELTA_CHAR(d) const int8_t d[3][3] = {{1,0,0},{0,1,0},{0,0,1}}
#define LEVI_CIVITA_CHAR(e) const int8_t e[3][3][3] =		\
    {{{0, 0, 0}, { 0, 0, 1}, { 0,-1, 0}},			\
     {{0, 0,-1}, { 0, 0, 0}, { 1, 0, 0}},			\
     {{0, 1, 0}, {-1, 0, 0}, { 0, 0, 0}}}

#define PI_DOUBLE(pi) const double pi = 3.1415926535897932385


__host__ int    is_bigendian(void);
__host__ double reverse_byte_order_double(char *);
__host__ int util_reverse_byte_order(void * arg, void * res, MPI_Datatype type);

__host__ __device__ double dot_product(const double a[3], const double b[3]);
__host__ __device__ void cross_product(const double a[3], const double b[3],
				       double result[3]);
__host__ __device__ double modulus(const double a[3]);
__host__ __device__ void   rotate_vector(double [3], const double [3]);

__host__ __device__ int    imin(const int i, const int j);
__host__ __device__ int    imax(const int i, const int j);
__host__ __device__ double dmin(const double a, const double b);
__host__ __device__ double dmax(const double a, const double b);

__host__ int util_jacobi(double a[3][3], double vals[3], double vecs[3][3]);
__host__ int util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]);
__host__ int util_discrete_volume_sphere(double r0[3], double a0, double * vn);
__host__ int util_gauss_jordan(const int n, double * a, double * b);
__host__ __device__ int util_dpythag(double a, double b, double * p);
__host__ int util_svd(int m, int n, double ** u, double * w, double ** v);
__host__ int util_svd_solve(int m, int n, double ** a, double * b, double * x);
__host__ int util_matrix_create(int m, int n, double *** p);
__host__ int util_vector_create(int m, double ** p);
__host__ int util_matrix_free(int m, double *** p);
__host__ int util_vector_free(double ** p);
__host__ int util_matrix_invert(int n, double ** a);

__host__ int util_random_unit_vector(int * state, double rhat[3]);
__host__ int util_ranlcg_reap_uniform(int * state, double * r);
__host__ int util_ranlcg_reap_gaussian(int * state, double r[2]);

#endif
