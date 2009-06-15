/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  $Id: d3q19.h,v 1.8.4.1 2009-06-15 11:25:20 cevi_parker Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#ifdef _D3Q19_

enum {NVEL = 19};
enum {LE_N_VEL_XING = 5};
enum {CVXBLOCK = 1};
enum {CVYBLOCK = 3};
enum {CVZBLOCK = 5};

extern const int cv[NVEL][3];
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
