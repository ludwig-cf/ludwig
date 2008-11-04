/****************************************************************************
 *
 *  util.h
 *
 *  $Id: util.h,v 1.1 2008-11-04 16:44:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) 
 *  
 ****************************************************************************/

#ifndef UTIL_H
#define UTIL_H

int is_bigendian(void);
double reverse_byte_order_double(char *);

#endif
