/*****************************************************************************
 *
 *  cs_limits.h
 *
 *  A container for a cubiodal region.
 *
 *  One may think of this as being associated with a flat address
 *  space starting at zero, and organised C-style (k running fastest).
 *
 *  We allow conversion of a flattened 1-d index iflat to 3-d (ic, jc, kc)
 *  and vice-versa.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

/*****************************************************************************
 *
 *  cs_limits_size
 *
 *  Return size, or volume, of region.
 *
 *****************************************************************************/

static inline int cs_limits_size(cs_limits_t lim) {

  assert(lim.imin <= lim.imax);
  assert(lim.jmin <= lim.jmax);
  assert(lim.kmin <= lim.kmax);

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

/*****************************************************************************
 *
 *  cs_limits_ic
 *
 *  x-coordinate from flat index.
 *
 *****************************************************************************/

static inline int cs_limits_ic(cs_limits_t lim, int iflat) {

  assert(0 <= iflat && iflat < cs_limits_size(lim));

  int ny   = 1 + lim.jmax - lim.jmin;
  int nz   = 1 + lim.kmax - lim.kmin;
  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  int ic = lim.imin + iflat/strx;

  assert(lim.imin <= ic && ic <= lim.imax);

  return ic;
}

/*****************************************************************************
 *
 *  cs_limits_jc
 *
 *  y-coordinate from flat index.
 *
 *****************************************************************************/

static inline int cs_limits_jc(cs_limits_t lim, int iflat) {

  assert(0 <= iflat && iflat < cs_limits_size(lim));

  int ny   = 1 + lim.jmax - lim.jmin;
  int nz   = 1 + lim.kmax - lim.kmin;
  int strz = 1;
  int stry = strz*nz;
  int strx = stry*ny;

  int jc = lim.jmin + (iflat % strx)/stry;

  assert(lim.jmin <= jc && jc <= lim.jmax);

  return jc;
}

/*****************************************************************************
 *
 *  cs_limits_kc
 *
 *  z-coordinate from flat index.
 *
 *****************************************************************************/

static inline int cs_limits_kc(cs_limits_t lim, int iflat) {

  assert(0 <= iflat && iflat < cs_limits_size(lim));

  int nz   = 1 + lim.kmax - lim.kmin;
  int strz = 1;
  int stry = strz*nz;

  int kc = lim.kmin + (iflat % stry)/strz;

  assert(lim.kmin <= kc && kc <= lim.kmax);

  return kc;
}

/*****************************************************************************
 *
 *  cs_limits_index
 *
 *  Flat index from (ic, jc, kc) triple.
 *
 *****************************************************************************/

static inline int cs_limits_index(cs_limits_t lim, int ic, int jc, int kc) {

  int iflat = -1;

  int ny  = 1 + lim.jmax - lim.jmin;
  int nz  = 1 + lim.kmax - lim.kmin;

  assert(lim.imin <= ic && ic <= lim.imax);
  assert(lim.jmin <= jc && jc <= lim.jmax);
  assert(lim.kmin <= kc && kc <= lim.kmax);

  iflat = (ic - lim.imin)*ny*nz + (jc - lim.jmin)*nz + (kc - lim.kmin);

  assert(0 <= iflat && iflat < cs_limits_size(lim));

  return iflat;
}

#endif
