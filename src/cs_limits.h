/*****************************************************************************
 *
 *  cs_limits.h
 *
 *  A container for a cubiodal region.
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

static inline int cs_limits_size(cs_limits_t lim) {

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

#endif
