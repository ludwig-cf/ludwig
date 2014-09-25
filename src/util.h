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

extern const double d_[3][3];
extern const double e_[3][3][3];
extern const double pi_;
extern const double r3_;

int    is_bigendian(void);
double reverse_byte_order_double(char *);
double dot_product(const double a[3], const double b[3]);
void   cross_product(const double a[3], const double b[3], double result[3]);
double modulus(const double a[3]);
void   rotate_vector(double [3], const double [3]);

int    imin(const int i, const int j);
int    imax(const int i, const int j);
double dmin(const double a, const double b);
double dmax(const double a, const double b);

int    util_jacobi(double a[3][3], double vals[3], double vecs[3][3]);
int    util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]);
int    util_discrete_volume_sphere(double r0[3], double a0, double * vn);
int    util_gauss_jordan(const int n, double * a, double * b);
int    util_dpythag(double a, double b, double * p);
int    util_svd(int m, int n, double ** u, double * w, double ** v);
int    util_svd_solve(int m, int n, double ** a, double * b, double * x);
int    util_matrix_create(int m, int n, double *** p);
int    util_vector_create(int m, double ** p);
int    util_matrix_free(int m, double *** p);
int    util_vector_free(double ** p);
int util_matrix_invert(int n, double ** a);

int util_ranlcg_reap_uniform(int * state, double * r);
int util_ranlcg_reap_gaussian(int * state, double r[2]);

#endif
