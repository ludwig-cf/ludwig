/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  $Id: d3q19.h,v 1.6.6.2 2008-06-25 18:19:50 erlend Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifdef _D3Q19_

enum {NVEL = 19};
enum {LE_N_VEL_XING = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

#ifdef _MPI_

enum {xcount = 1};
extern MPI_Datatype xtypes[xcount];
extern int xblocklens[xcount];
extern MPI_Aint xdisp_right[xcount];
extern MPI_Aint xdisp_left[xcount];

enum {ycount = 1};
extern MPI_Datatype ytypes[ycount];
extern int yblocklens[ycount];
extern MPI_Aint ydisp_right[ycount];
extern MPI_Aint ydisp_left[ycount];

enum {zcount = 1};
extern MPI_Datatype ztypes[zcount];
extern int zblocklens[zcount];
extern MPI_Aint zdisp_right[zcount];
extern MPI_Aint zdisp_left[zcount];

#endif _MPI_

#endif
