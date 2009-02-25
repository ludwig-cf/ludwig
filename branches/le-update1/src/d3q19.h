/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  $Id: d3q19.h,v 1.7 2008-08-25 18:13:55 kevin Exp $
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
enum {xcountcv = 1};
enum {ycountcv = 3};
enum {zcountcv = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

extern const int xblocklens_cv[xcountcv];
extern const int xdisp_fwd_cv[xcountcv];
extern const int xdisp_bwd_cv[xcountcv];

extern const int yblocklens_cv[ycountcv];
extern const int ydisp_fwd_cv[ycountcv];
extern const int ydisp_bwd_cv[ycountcv];

extern const int zblocklens_cv[zcountcv];
extern const int zdisp_fwd_cv[zcountcv];
extern const int zdisp_bwd_cv[zcountcv];

#endif
