/****************************************************************************
 *
 *  util.h
 *
 *  $Id: util.h,v 1.2 2009-05-15 09:10:37 kevin Exp $
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

#endif
