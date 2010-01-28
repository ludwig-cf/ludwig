/****************************************************************************
 *
 *  util.h
 *
 *  $Id: util.h,v 1.2.6.1 2009-11-04 10:22:25 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  
 ****************************************************************************/

#ifndef UTIL_H
#define UTIL_H

extern const double d_[3][3];
extern const double e_[3][3][3];

int is_bigendian(void);
double reverse_byte_order_double(char *);
double dot_product(const double a[3], const double b[3]);

#endif
