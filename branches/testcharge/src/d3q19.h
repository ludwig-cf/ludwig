/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel computing Centre
 *
 *  (c) 2008-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ronojoy Adhikari computed this D3Q19 basis.
 *
 *****************************************************************************/

#ifndef D3Q19_MODEL_H
#define D3Q19_MODEL_H

enum {NDIM = 3};
enum {NVEL = 19};
enum {CVXBLOCK = 1};
enum {CVYBLOCK = 3};
enum {CVZBLOCK = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

extern const int xblocklen_cv[CVXBLOCK];
extern const int xdisp_fwd_cv[CVXBLOCK];
extern const int xdisp_bwd_cv[CVXBLOCK];

extern const int yblocklen_cv[CVYBLOCK];
extern const int ydisp_fwd_cv[CVYBLOCK];
extern const int ydisp_bwd_cv[CVYBLOCK];

extern const int zblocklen_cv[CVZBLOCK];
extern const int zdisp_fwd_cv[CVZBLOCK];
extern const int zdisp_bwd_cv[CVZBLOCK];

#endif
