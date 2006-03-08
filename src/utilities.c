#include "utilities.h"

/*****************************************************************************
 *
 *  UTIL_fdistance_sq
 *
 *  Distance between two points with position vectors r1 nad r2
 *  is returned as a Float. Squared.
 *
 *****************************************************************************/

Float UTIL_fdistance_sq(FVector r1, FVector r2) {

  Float sq = 0.0;

  sq += (r1.x - r2.x)*(r1.x - r2.x);
  sq += (r1.y - r2.y)*(r1.y - r2.y);
  sq += (r1.z - r2.z)*(r1.z - r2.z);

  return sq;
}

/*****************************************************************************
 *
 *  UTIL_fvector_mod
 *
 *  Return the length of a given vector.
 *
 *****************************************************************************/

Float UTIL_fvector_mod(FVector r) {

  Float rmod;

  rmod = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);

  return rmod;
}


/*****************************************************************************
 *
 *  UTIL_fvector_zero
 *
 *  Return zero.
 *
 *****************************************************************************/

FVector UTIL_fvector_zero() {

  FVector zero = {0.0, 0.0, 0.0};

  return zero;
}

/*****************************************************************************
 *
 *  UTIL_dot_product
 *
 *  Scalar product of two FVectors.
 *
 *****************************************************************************/

Float UTIL_dot_product(FVector v1, FVector v2) {

  return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}


/*****************************************************************************
 *
 *  UTIL_fvector_add
 *
 *  A vector addition.
 *
 *****************************************************************************/

FVector UTIL_fvector_add(FVector v1, FVector v2) {

  FVector v;

  v.x = v1.x + v2.x;
  v.y = v1.y + v2.y;
  v.z = v1.z + v2.z;

  return v;
}


/*****************************************************************************
 *
 *  UTIL_fvector_subtract
 *
 *  A vector subtraction.
 *
 *****************************************************************************/

FVector UTIL_fvector_subtract(FVector v1, FVector v2) {

  FVector v;

  v.x = v1.x - v2.x;
  v.y = v1.y - v2.y;
  v.z = v1.z - v2.z;

  return v;
}

/*****************************************************************************
 *
 *  UTIL_cross_product
 *
 *  Vector product of two FVectors. Order is important!
 *
 *****************************************************************************/

FVector UTIL_cross_product(FVector v1, FVector v2) {

  FVector result;

  result.x = v1.y*v2.z - v1.z*v2.y;
  result.y = v1.z*v2.x - v1.x*v2.z;
  result.z = v1.x*v2.y - v1.y*v2.x;

  return result;
}

/*****************************************************************************
 *
 *  UTIL_rotate_vector
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

FVector UTIL_rotate_vector(FVector v, FVector w) {

  FVector vrot;
  FVector what;
  double  theta, ct, st;
  double  vdotw;

  theta = sqrt(w.x*w.x + w.y*w.y + w.z*w.z);

  if (theta == 0.0) {
    /* There no rotation. */
    vrot.x = v.x;
    vrot.y = v.y;
    vrot.z = v.z;
  }
  else {
    /* Work out the unit axis of rotation */

    what.x = w.x / theta;
    what.y = w.y / theta;
    what.z = w.z / theta;

    /* Rotation */

    st = sin(theta);
    ct = cos(theta);
    vdotw = v.x*what.x + v.y*what.y + v.z*what.z;

    vrot.x = (1.0 - ct)*vdotw*what.x + ct*v.x + st*(what.y*v.z - what.z*v.y);
    vrot.y = (1.0 - ct)*vdotw*what.y + ct*v.y + st*(what.z*v.x - what.x*v.z);
    vrot.z = (1.0 - ct)*vdotw*what.z + ct*v.z + st*(what.x*v.y - what.y*v.x);
  }

  return vrot;
}
