/*****************************************************************************
 *
 *  util_math_inline.h
 *
 *  __host__ __device__ static inline utilities.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_MATH_INLINE_H_
#define LUDWIG_UTIL_MATH_INLINE_H_

#define HDSI_ __host__ __device__ static inline

/*****************************************************************************
 *
 *  util_imax
 *
 *  integer max()
 *
 *****************************************************************************/

HDSI_ int util_imax(int a, int b) {

  return (a > b) ? a : b;
}

/*****************************************************************************
 *
 *  util_imin
 *
 *  integer min()
 *
 *****************************************************************************/

HDSI_ int util_imin(int a, int b) {

  return (a < b) ? a : b;
}

/*****************************************************************************
 *
 *  util_square_modulus_int8
 *
 *  Return as int (sic).
 *
 *****************************************************************************/

HDSI_ int util_square_modulus_int8(const int8_t cv[3]) {

  return (int) (cv[0]*cv[0] + cv[1]*cv[1] + cv[2]*cv[2]);
}

#undef HDSI_

#endif
