/*****************************************************************************
 *
 *  util.c
 *
 *  Utility functions.
 *
 *  Little / big endian stuff based on suggestions by Harsha S.
 *  Adiga from IBM.
 *
 *  $Id: util.c,v 1.2 2009-05-15 09:10:37 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh (2008)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "util.h"

#define c0 0.0
#define c1 1.0

const double d_[3][3]    = {{c1, c0, c0}, {c0, c1, c0}, {c0, c0, c1}};
const double e_[3][3][3] = {{{c0, c0, c0}, { c0, c0, c1}, {c0,-c1, c0}},
			    {{c0, c0,-c1}, { c0, c0, c0}, {c1, c0, c0}},
			    {{c0, c1, c0}, {-c1, c0, c0}, {c0, c0, c0}}}; 


/***************************************************************************
 *
 *  is_bigendian
 *
 *  Byte order for this 4-byte int is 00 00 00 01 for big endian (most
 *  significant byte stored first).
 *
 ***************************************************************************/

int is_bigendian() {

  const int i = 1;

  return (*(char *) &i == 0);
}

/****************************************************************************
 *
 *  reverse_byte_order_double
 *
 *  Reverse the bytes in the char argument to make a double.
 *
 *****************************************************************************/

double reverse_byte_order_double(char * c) {

  double result;
  char * p = (char *) &result;
  int b;

  for (b = 0; b < sizeof(double); b++) {
    p[b] = c[sizeof(double) - (b + 1)];
  }

  return result;
}
