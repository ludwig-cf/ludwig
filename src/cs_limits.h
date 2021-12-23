/*****************************************************************************
 *
 *  cs_limits.h
 *
 *  A container (only) for a cubiodal region.
 *
 *****************************************************************************/

#ifndef LUDWIG_CS_LIMITS_H
#define LUDWIG_CS_LIMITS_H

typedef struct cs_limits_s cs_limits_t;

struct cs_limits_s {
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
};

#endif
