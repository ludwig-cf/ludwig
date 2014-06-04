/****************************************************************************
 *
 *  util.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  (c) 2010 The University of Edinburgh
 *  
 ****************************************************************************/

#ifndef UTIL_H
#define UTIL_H

#ifndef __NVCC__
#define __extern_c__
#else
#define __extern_c__ extern "C"
#endif

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

__extern_c__ int util_matrix_create(int mrow, int ncol, double *** a);
__extern_c__ int util_matrix_free(int mrow, double *** a);
__extern_c__ int util_matrix_invert(int n, double ** a);

#endif
