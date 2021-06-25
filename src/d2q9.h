/*****************************************************************************
 *
 *  d2q9.h
 *
 *  D2Q9 definitions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_D2Q9_MODEL_H
#define LUDWIG_D2Q9_MODEL_H

enum {NDIM9 = 2};
enum {NVEL9 = 9};
enum {CVXBLOCK9 = 1};
enum {CVYBLOCK9 = 3};
enum {CVZBLOCK9 = 1};

#ifdef _D2Q9_

extern const    int cv[NVEL9][3];
extern const double wv[NVEL9];
extern const double norm_[NVEL9];
extern const double ma_[NVEL9][NVEL9];
extern const double mi_[NVEL9][NVEL9];

extern const int xblocklen_cv[CVXBLOCK9];
extern const int xdisp_fwd_cv[CVXBLOCK9];
extern const int xdisp_bwd_cv[CVXBLOCK9];

extern const int yblocklen_cv[CVYBLOCK9];
extern const int ydisp_fwd_cv[CVYBLOCK9];
extern const int ydisp_bwd_cv[CVYBLOCK9];

extern const int zblocklen_cv[CVZBLOCK9];
extern const int zdisp_fwd_cv[CVZBLOCK9];
extern const int zdisp_bwd_cv[CVZBLOCK9];

#define LB_NORMALISERS_DOUBLE(norm) const double norm[NVEL9] = { \
    1.0, 3.0, 3.0, 9.0/2.0, 9.0, 9.0/2.0, 1.0/4.0, 3.0/8.0, 3.0/8.0};

#endif

#endif
