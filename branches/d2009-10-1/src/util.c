/*****************************************************************************
 *
 *  util.c
 *
 *  Utility functions, including vectors.
 *
 *  Little / big endian stuff based on suggestions by Harsha S.
 *  Adiga from IBM.
 *
 *  $Id: util.c,v 1.2.6.2 2010-06-02 13:57:25 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "util.h"

#define c0 0.0
#define c1 1.0

const double pi_ = 3.1415926535897932385;
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

/*****************************************************************************
 *
 *  dot_product
 *
 *****************************************************************************/

double dot_product(const double a[3], const double b[3]) {

  return (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z]);
}

/*****************************************************************************
 *
 *  cross_product
 *
 *****************************************************************************/

void cross_product(const double a[3], const double b[3], double result[3]) {

  result[X] = a[Y]*b[Z] - a[Z]*b[Y];
  result[Y] = a[Z]*b[X] - a[X]*b[Z];
  result[Z] = a[X]*b[Y] - a[Y]*b[X];

  return;
}

/*****************************************************************************
 *
 *  modulus
 *
 *****************************************************************************/

double modulus(const double a[3]) {

  return sqrt(a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z]);
}

/*****************************************************************************
 *
 *  rotate_vector
 *
 *  Rotate the vector v around the unit axis of rotation \hat{w}
 *  by an angle of \theta, where \theta = |w|. (For example, w
 *  might be an angular velocity.)
 *
 *  The rotated vector is computed via
 *      v' = (1 - cos \theta)(\hat{w}.v) \hat{w} + cos \theta v +
 *           (\hat{w} x v) sin \theta      
 *
 *  For theta positive this gives rotations in the correct sense
 *  in the right-handed coordinate system.
 *
 ****************************************************************************/

void rotate_vector(double v[3], const double w[3]) {

  double what[3], vrot[3];
  double theta, ct, st;
  double vdotw;

  theta = sqrt(w[X]*w[X] + w[Y]*w[Y] + w[Z]*w[Z]);

  if (theta == 0.0) {
    /* There is no rotation. */
   }
  else {
    /* Work out the unit axis of rotation */

    what[X] = w[X] / theta;
    what[Y] = w[Y] / theta;
    what[Z] = w[Z] / theta;

    /* Rotation */

    st = sin(theta);
    ct = cos(theta);
    vdotw = v[X]*what[X] + v[Y]*what[Y] + v[Z]*what[Z];

    vrot[X] = ct*v[X] + st*(what[Y]*v[Z] - what[Z]*v[Y]);
    vrot[Y] = ct*v[Y] + st*(what[Z]*v[X] - what[X]*v[Z]);
    vrot[Z] = ct*v[Z] + st*(what[X]*v[Y] - what[Y]*v[X]);
    v[X] = (1.0 - ct)*vdotw*what[X] + vrot[X];
    v[Y] = (1.0 - ct)*vdotw*what[Y] + vrot[Y];
    v[Z] = (1.0 - ct)*vdotw*what[Z] + vrot[Z];
  }

  return;
}

/*****************************************************************************
 *
 *  imin, imax, dmin, dmax
 *
 *  minimax functions
 *
 *****************************************************************************/

int imin(const int i, const int j) {
  return ((i < j) ? i : j);
}

int imax(const int i, const int j) {
  return ((i > j) ? i : j);
}

double dmin(const double a, const double b) {
  return ((a < b) ? a : b);
}

double dmax(const double a, const double b) {
  return ((a > b) ? a : b);
}
