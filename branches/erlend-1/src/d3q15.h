/*****************************************************************************
 *
 *  d3q15.h
 *
 *  D3Q15 definitions.
 *
 *  $Id: d3q15.h,v 1.5.6.6 2008-07-25 17:08:35 erlend Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifdef _D3Q15_

enum {NVEL = 15};
enum {LE_N_VEL_XING = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

#ifdef _MPI_

enum {xcountcv = 1};
extern int xblocklens_cv[xcountcv];
extern int xdisp_fwd_cv[xcountcv];
extern int xdisp_bwd_cv[xcountcv];

enum {ycountcv = 3};
extern int yblocklens_cv[ycountcv];
extern int ydisp_fwd_cv[ycountcv];
extern int ydisp_bwd_cv[ycountcv];

enum {zcountcv = 5};
extern int zblocklens_cv[zcountcv];
extern int zdisp_fwd_cv[zcountcv];
extern int zdisp_bwd_cv[zcountcv];

#endif

#endif

